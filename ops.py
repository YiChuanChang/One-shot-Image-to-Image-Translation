import tensorflow as tf
import tensorflow.contrib as tf_contrib
import tensorflow.contrib.slim as slim
from custom_vgg16 import *


weight_init = tf_contrib.layers.variance_scaling_initializer() # kaming init for encoder / decoder
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
	# pytorch alpha is 0.01
	return tf.nn.leaky_relu(x, alpha)

def relu(x):
	return tf.nn.relu(x)

def tanh(x):
	return tf.tanh(x)
##################################################################################
# Normalization function
##################################################################################
def nearest_patch_swapping(content_features, style_features, patch_size=3):
	# channels for both the content and style, must be the same
	c_shape = tf.shape(content_features)
	s_shape = tf.shape(style_features)
	channel_assertion = tf.Assert(
		tf.equal(c_shape[3], s_shape[3]), ['number of channels  must be the same'])

	with tf.control_dependencies([channel_assertion]):
		# spatial shapes for style and content features
		c_height, c_width, c_channel = c_shape[1], c_shape[2], c_shape[3]

		# convert the style features into convolutional kernels
		style_kernels = tf.extract_image_patches(
			style_features, ksizes=[1, patch_size, patch_size, 1],
			strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
		style_kernels = tf.squeeze(style_kernels, axis=0)
		style_kernels = tf.transpose(style_kernels, perm=[2, 0, 1])

		# gather the conv and deconv kernels
		v_height, v_width = style_kernels.get_shape().as_list()[1:3]
		deconv_kernels = tf.reshape(
			style_kernels, shape=(patch_size, patch_size, c_channel, v_height*v_width))

		kernels_norm = tf.norm(style_kernels, axis=0, keep_dims=True)
		kernels_norm = tf.reshape(kernels_norm, shape=(1, 1, 1, v_height*v_width))

		# calculate the normalization factor
		mask = tf.ones((c_height, c_width), tf.float32)
		fullmask = tf.zeros((c_height+patch_size-1, c_width+patch_size-1), tf.float32)
		for x in range(patch_size):
			for y in range(patch_size):
				paddings = [[x, patch_size-x-1], [y, patch_size-y-1]]
				padded_mask = tf.pad(mask, paddings=paddings, mode="CONSTANT")
				fullmask += padded_mask
		pad_width = int((patch_size-1)/2)
		deconv_norm = tf.slice(fullmask, [pad_width, pad_width], [c_height, c_width])
		deconv_norm = tf.reshape(deconv_norm, shape=(1, c_height, c_width, 1))

		########################
		# starting convolution #
		########################
		# padding operation
		pad_total = patch_size - 1
		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg
		paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]

		# convolutional operations
		net = tf.pad(content_features, paddings=paddings, mode="REFLECT")
		net = tf.nn.conv2d(
			net,
			tf.div(deconv_kernels, kernels_norm+1e-7),
			strides=[1, 1, 1, 1],
			padding='VALID')
		# find the maximum locations
		best_match_ids = tf.argmax(net, axis=3)
		best_match_ids = tf.cast(
			tf.one_hot(best_match_ids, depth=v_height*v_width), dtype=tf.float32)

		# find the patches and warping the output
		unnormalized_output = tf.nn.conv2d_transpose(
			value=best_match_ids,
			filter=deconv_kernels,
			output_shape=(c_shape[0], c_height+pad_total, c_width+pad_total, c_channel),
			strides=[1, 1, 1, 1],
			padding='VALID')
		unnormalized_output = tf.slice(unnormalized_output, [0, pad_beg, pad_beg, 0], c_shape)
		output = tf.div(unnormalized_output, deconv_norm)
		output = tf.reshape(output, shape=c_shape)

		# output the swapped feature maps
		return output

def adain_normalization(features):
	epsilon = 1e-7
	mean_features, colorization_kernels = tf.nn.moments(features, [1, 2], keep_dims=True)
	normalized_features = tf.div(
		tf.subtract(features, mean_features), tf.sqrt(tf.add(colorization_kernels, epsilon)))
	return normalized_features, colorization_kernels, mean_features

def avatar_norm(style, content, patch_size=3, ratio=1.0):
    # feature projection (AdaIn)
	projected_content_features, _, _ = \
		adain_normalization(content)
	projected_style_features, style_kernels, mean_style_features = \
		adain_normalization(style)

	# feature rearrangement 
	rearranged_features = nearest_patch_swapping(
		projected_content_features, projected_style_features, patch_size=patch_size)
	rearranged_features = ratio * rearranged_features + \
		(1 - ratio) * projected_content_features

	# feature reconstruction (AdaIn)
	reconstructed_features = tf.sqrt(style_kernels) * rearranged_features + mean_style_features
	
	return reconstructed_features


def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
	# gamma, beta = style_mean, style_std from MLP

	c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
	c_std = tf.sqrt(c_var + epsilon)

	return gamma * ((content - c_mean) / c_std) + beta


def instance_norm(x, scope='instance_norm'):
	return tf_contrib.layers.instance_norm(x,
											epsilon=1e-05,
											center=True, scale=True,
											scope=scope)

def layer_norm(x, scope='layer_norm') :
	return tf_contrib.layers.layer_norm(x,
										center=True, scale=True,
										scope=scope)
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def AdaIn(content, style, alpha=1, epsilon=1e-5):
	content_mean, content_variance = tf.nn.moments(content, [1,2], keep_dims=True)
	style_mean, style_variance = tf.nn.moments(style, [1,2], keep_dims=True)
	normalized_content = tf.nn.batch_normalization(content, content_mean, \
						content_variance, style_mean, tf.sqrt(style_variance), epsilon)
	normalized_content = alpha*normalized_content + (1-alpha)*content

	return normalized_content

##################################################################################
# Sampling
##################################################################################

def down_sample(x) :
	return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

def up_sample(x, scale_factor=2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.image.resize_nearest_neighbor(x, size=new_size)

def adaptive_avg_pooling(x):
	# global average pooling
	gap = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

	return gap

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock'):
	with tf.variable_scope(scope):
		with tf.variable_scope('res1'):
			x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
			x = instance_norm(x)
			x = relu(x)

		with tf.variable_scope('res2'):
			x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
			x = instance_norm(x)

		return x + x_init

def adaptive_resblock(x_init, channels, mu, sigma, use_bias=True, scope='adaptive_resblock') :
	with tf.variable_scope(scope):
		with tf.variable_scope('res1'):
			x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
			x = adaptive_instance_norm(x, mu, sigma)
			x = relu(x)

		with tf.variable_scope('res2'):
			x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
			x = adaptive_instance_norm(x, mu, sigma)

		return x + x_init

##################################################################################
# Non Local Block
##################################################################################

def NonLocalBlock(input_x, out_channels, sub_sample=True, is_bn=True, scope='NonLocalBlock'):
    batchsize, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('g') as scope:
            g = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='g')
            if sub_sample:
                g = slim.max_pool2d(g, [2,2], stride=2, scope='g_max_pool')

        with tf.variable_scope('phi') as scope:
            phi = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='phi')
            if sub_sample:
                phi = slim.max_pool2d(phi, [2,2], stride=2, scope='phi_max_pool')

        with tf.variable_scope('theta') as scope:
            theta = slim.conv2d(input_x, out_channels, [1,1], stride=1, scope='theta')

        g_x = tf.reshape(g, [batchsize,out_channels, -1])
        g_x = tf.transpose(g_x, [0,2,1])

        theta_x = tf.reshape(theta, [batchsize, out_channels, -1])
        theta_x = tf.transpose(theta_x, [0,2,1])
        phi_x = tf.reshape(phi, [batchsize, out_channels, -1])

        f = tf.matmul(theta_x, phi_x)
        # ???
        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)
        y = tf.reshape(y, [batchsize, height, width, out_channels])
        with tf.variable_scope('w') as scope:
            w_y = slim.conv2d(y, in_channels, [1,1], stride=1, scope='w')
            if is_bn:
                w_y = slim.batch_norm(w_y)
        z = input_x + w_y
        return z
##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv'):
	with tf.variable_scope(scope):
		if scope.__contains__("discriminator") :
			weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
		else :
			weight_init = tf_contrib.layers.variance_scaling_initializer()

		if pad_type == 'zero' :
			x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
		if pad_type == 'reflect' :
			x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
		if sn:
			w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
            					regularizer=weight_regularizer)
			bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
			x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
							strides=[1, stride, stride, 1], padding='VALID')
			if use_bias :
				x = tf.nn.bias_add(x, bias)
		else:
			x = tf.layers.conv2d(inputs=x, filters=channels,
								kernel_size=kernel, kernel_initializer=weight_init,
								kernel_regularizer=weight_regularizer,
								strides=stride, use_bias=use_bias)

		return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Calculate Loss
##################################################################################
def L2_loss(x, y):
	return tf.losses.mean_squared_error(x, y)

def L1_loss(x, y):
	loss = tf.reduce_mean(tf.abs(x - y))

	return loss

def generator_loss(fake):
	# LS Gan
	n_scale = len(fake)
	loss = []
	fake_loss = 0
	for i in range(n_scale) :
		fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))
		loss.append(fake_loss)


	return sum(loss) # sum up different scale loss

def discriminator_loss(real, fake):
	n_scale = len(real)
	loss = []

	real_loss = 0
	fake_loss = 0

	for i in range(n_scale) :
		real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
		fake_loss = tf.reduce_mean(tf.square(fake[i]))

		loss.append(real_loss + fake_loss)

	return sum(loss)

# gram matrix per layer
def gram_matrix(x):
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, h*w, ch])
    # gram = tf.batch_matmul(features, features, adj_x=True)/tf.constant(ch*w*h, tf.float32)
    gram = tf.matmul(features, features, adjoint_a=True)/tf.constant(ch*w*h, tf.float32)
    return gram

def perceptual_loss_style(image_A, image_B, vgg_weight, batchsize):

	vgg_A = custom_Vgg16(image_A, data_dict=vgg_weight)
	feature_A = [vgg_A.conv1_2, vgg_A.conv2_2, vgg_A.conv3_3, vgg_A.conv4_3, vgg_A.conv5_3]
	gram_A = [gram_matrix(l) for l in feature_A]

	vgg_B = custom_Vgg16(image_B, data_dict=vgg_weight)
	feature_B = [vgg_B.conv1_2, vgg_B.conv2_2, vgg_B.conv3_3, vgg_B.conv4_3, vgg_B.conv5_3]
	gram_B = [gram_matrix(l) for l in feature_B]

	# compute style loss
	loss_s = tf.zeros(batchsize, tf.float32)
	for g_A, g_B in zip(gram_A, gram_B):
		loss_s += tf.reduce_mean(tf.subtract(g_A, g_B) ** 2, [1, 2])

	return tf.squeeze(loss_s)


def perceptual_loss_content(image_A, image_B, vgg_weight, batchsize):
	vgg_A = custom_Vgg16(image_A, data_dict=vgg_weight)
	feature_A = [vgg_A.conv3_1, vgg_A.conv4_1]#[vgg_A.conv1_1, vgg_A.conv2_1]

	vgg_B = custom_Vgg16(image_B, data_dict=vgg_weight)
	feature_B = [vgg_B.conv3_1, vgg_B.conv4_1]#[vgg_B.conv1_1, vgg_B.conv2_1, 

	# compute feature loss
	loss_f = tf.zeros(batchsize, tf.float32)
	for f_A, f_B in zip(feature_A, feature_B):
		loss_f += tf.reduce_mean(tf.subtract(f_A, f_B) ** 2, [1, 2, 3])

	return tf.squeeze(loss_f)

def compute_total_variation_loss(inputs, weights=1, scope=None):
	"""compute the total variation loss L1 norm"""
	inputs_shape = tf.shape(inputs)
	height = inputs_shape[1]
	width = inputs_shape[2]

	with tf.variable_scope(scope, 'total_variation_loss', [inputs]):
		loss_y = tf.losses.absolute_difference(
			tf.slice(inputs, [0, 0, 0, 0], [-1, height-1, -1, -1]),
			tf.slice(inputs, [0, 1, 0, 0], [-1, -1, -1, -1]),
			weights=weights,
			scope='loss_y')
		loss_x = tf.losses.absolute_difference(
			tf.slice(inputs, [0, 0, 0, 0], [-1, -1, width-1, -1]),
			tf.slice(inputs, [0, 0, 1, 0], [-1, -1, -1, -1]),
			weights=weights,
			scope='loss_x')
		loss = loss_y + loss_x
		return loss


