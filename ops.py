import tensorflow as tf
import tensorflow.contrib as tf_contrib
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
def L2_loss(v, eps=1e-12):
	return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

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
	gram_B = [gram_matrix(l) for l in feature_A]

	# compute style loss
	loss_s = tf.zeros(batchsize, tf.float32)
	for g_A, g_B in zip(gram_A, gram_B):
		loss_s += tf.reduce_mean(tf.subtract(g_A, g_B) ** 2, [1, 2])

	return tf.squeeze(loss_s)


def perceptual_loss_content(image_A, image_B, vgg_weight, batchsize):
	vgg_A = custom_Vgg16(image_A, data_dict=vgg_weight)
	feature_A = [vgg_A.conv1_2, vgg_A.conv2_2, vgg_A.conv3_3, vgg_A.conv4_3, vgg_A.conv5_3]

	vgg_B = custom_Vgg16(image_B, data_dict=vgg_weight)
	feature_B = [vgg_B.conv1_2, vgg_B.conv2_2, vgg_B.conv3_3, vgg_B.conv4_3, vgg_B.conv5_3]

	# compute feature loss
	loss_f = tf.zeros(batchsize, tf.float32)
	for f_A, f_B in zip(feature_A, feature_B):
		loss_f += tf.reduce_mean(tf.subtract(f_A, f_B) ** 2, [1, 2, 3])

	return tf.squeeze(loss_f)


