import tensorflow as tf
import time
from utils import *
from ops import *
from glob import glob
from tensorflow.contrib.data import batch_and_drop_remainder


class ZSIT(object):
	def __init__(self, sess, args):
		self.model_name = 'ZSIT'
		self.sess = sess

		# directory
		self.checkpoint_dir = args.checkpoint_dir
		self.result_dir = args.result_dir # for saving image result when training
		self.log_dir = args.log_dir # for saving summary
		self.sample_dir = args.sample_dir # for saveing images result when training

		# traning setting

		self.dataset_name = args.dataset
		self.trainA_dataset = glob('./dataset/{}/*.jpg'.format(self.dataset_name + '/trainA'))
		self.trainB_dataset = glob('./dataset/{}/*.jpg'.format(self.dataset_name + '/trainB'))
		self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

		self.augment_flag = args.augment_flag
		self.epoch = args.epoch
		self.iteration = args.iteration
		self.batch_size = args.batch_size
		self.print_freq = args.print_freq
		self.save_freq = args.save_freq
		self.init_lr = args.lr

		# image infomation
		self.img_h = args.img_h
		self.img_w = args.img_w
		self.img_c = args.img_c

		# weight for loss ratio
		self.gan_w = args.gan_w # for loss from discriminator
		self.con_w = args.con_w # for loss: L1(origin_content, reconstruct_content)
		self.sty_w = args.sty_w # for loss: L1(origin_style, reconstruct_style)
		self.cyc_w = args.cyc_w # for loss: cycle reconstruct
		self.rec_w = args.rec_w # for loss: reconstruct original image

		self.ch = args.ch

		# content Encoder Setting
		self.n_cont_res = args.n_cont_res
		self.n_cont_downsample = args.n_cont_downsample # conv layer number
		self.useUNet = args.useUNet # use UNet or not

		# style Encoder setting
		self.n_style_res = args.n_style_res
		self.n_style_downsample = args.n_style_downsample # conv layer number


		# Decoder setting
		self.n_upsample = args.n_upsample

		# Discriminator setting
		self.n_scale = args.n_scale
		self.n_dis = args.n_dis # discirminator layer number
		self.sn = args.sn # use spetral norm or not

	#######################################################################
	# Encoder and Decoders
	#######################################################################

	def style_encoder(self, x, reuse=False, scope='style_encoder'):
		channel = self.ch
		with tf.variable_scope(scope, reuse=reuse):
			x = conv(x, channel, kernel=7, stride=1, pad=3, \
				pad_type='reflect', scope='conv_0')
			x = instance_norm(x, scope='ins_0')
			x = relu(x)

			for i in range(self.n_style_downsample):
				x = conv(x, channel*2, kernel=4, stride=2, pad=1, \
					pad_type='reflect', scope='conv_'+str(i+1))
				x = instance_norm(x, scope='ins_'+str(i+1))
				x = relu(x)
				channel = channel*2

			for i in range(self.n_style_res):
				# resblock consists of instance_norm and relu
				x = resblock(x, channel, scope='resblock_'+str(i))

			return x


	def content_encoder(self, x, reuse=False, scope='content_encoder'):
		channel = self.ch 
		content_layers = []
		with tf.variable_scope(scope, reuse=reuse):
			x = conv(x, channel, kernel=7, stride=1, pad=3, \
				pad_type='reflect', scope='conv_0')
			x = instance_norm(x, scope='ins_0')
			content_layers.append(x)
			x = relu(x)

			for i in range(self.n_cont_downsample):
				x = conv(x, channel*2, kernel=4, stride=2, pad=1, \
					pad_type='reflect', scope='conv_'+str(i+1))
				x = instance_norm(x, scope='ins_'+str(i+1))
				content_layers.append(x)
				x = relu(x)
				channel = channel*2

			for i in range(self.n_cont_res):
				# resblock consists of instance_norm and relu
				x = resblock(x, channel, scope='resblock_'+str(i))

			return x, content_layers

	def decoder(self, style, content, content_layers, reuse=False, scope='decoder'):
		channel = self.ch
		with tf.variable_scope(scope, reuse=reuse):
			if(self.useUNet):
				x = tf.concat([content, style, content_layers[self.n_cont_downsample]], 3)
			else:
				x = tf.concat([content, style], 3)

			for i in range(self.n_upsample):
				x = up_sample(x, scale_factor = 2)
				x = conv(x, channel//2, kernel=5, stride=1, pad=2, \
					pad_type='reflect', scope='conv_'+str(i))
				x = instance_norm(x, scope='ins_'+str(i))
				if(self.useUNet):
					x = tf.concat([x, content_layers[self.n_cont_downsample-i-1]], 3)
				x = relu(x)
				channel = channel//2
			x = conv(x, channels=self.img_c, kernel=7, stride=1, \
				pad=3, pad_type='reflect', scope='G_logit')
			x = tanh(x)

			return x

	# def UNet(self):
		

	##################################################################################
	# discriminator
	##################################################################################

	def scale_discriminator(self, x_init, reuse=False, scope='scale_discriminator'):
		D_logit = []
		with tf.variable_scope(scope, reuse=reuse):
			for scale in range(self.n_scale):
				channel = self.ch
				x = conv(x_init, channel, kernel=4, sn=self.sn, \
					stride=2, pad=1, pad_type='reflect', \
					scope='scale_'+str(scale)+'conv_0')
				x = lrelu(x, 0.2)
				for i in range(1, self.n_dis):
					x = conv(x, channel*2, kernel=4, sn=self.sn, \
						stride=2, pad=1, pad_type='reflect', \
						scope='scale_'+str(scale)+'conv_'+str(i))
					channel = channel*2

				x = conv(x, channels=1, kernel=1, sn=self.sn, \
					stride=1, scope='scale_'+str(scale)+'_D_logit')
				D_logit.append(x)

				x_init = down_sample(x_init)

			return D_logit
	
	##################################################################################
	# Network
	##################################################################################
	def Encoder(self, x, reuse=False):
		style = self.style_encoder(x, reuse=reuse, scope='style_encoder')
		content, content_layers = self.content_encoder(x, reuse=reuse, scope='content_encoder')
		return style, content, content_layers

	def Decoder(self, style, content, content_layers, reuse=False):
		x = self.decoder(style, content, content_layers, reuse=reuse, scope='decoder')
		return x

	def Discriminator_real(self, A, B):
		real_A_logit = self.scale_discriminator(A, scope="discriminator_A")
		real_B_logit = self.scale_discriminator(B, scope="discriminator_B")

		return real_A_logit, real_B_logit

	def Discriminator_fake(self, A, B):
		fake_A_logit = self.scale_discriminator(A, reuse=True, scope="discriminator_A")
		fake_B_logit = self.scale_discriminator(B, reuse=True, scope="discriminator_B")

		return fake_A_logit, fake_B_logit


	def build_network(self):
		self.lr = tf.placeholder(tf.float32, name='learning_rate')

		''' Get Data '''
		Image_Data_Class = ImageData(self.img_h, self.img_w, self.img_c, \
			self.augment_flag)
		trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
		trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

		trainA = trainA.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()
		trainB = trainB.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=8).apply(batch_and_drop_remainder(self.batch_size)).repeat()

		trainA_iterator = trainA.make_one_shot_iterator()
		trainB_iterator = trainB.make_one_shot_iterator()


		self.domain_A = trainA_iterator.get_next()
		self.domain_B = trainB_iterator.get_next()

		''' Define encoder, decoder, discriminator '''
		# encoding
		style_A, content_A, content_A_layers = self.Encoder(self.domain_A)
		style_B, content_B, content_B_layers = self.Encoder(self.domain_B, reuse=True)
		# decoding (same domain)
		sty_A_con_A = self.Decoder(style_A, content_A, content_A_layers)
		sty_B_con_B = self.Decoder(style_B, content_B, content_B_layers, reuse=True)
		# decoding (cross domain)
		sty_A_con_B = self.Decoder(style_A, content_B, content_B_layers, reuse=True)
		sty_B_con_A = self.Decoder(style_B, content_A, content_A_layers, reuse=True)

		# re-encoding
		style_A_re, content_B_re, content_B_re_layers = self.Encoder(sty_A_con_B, reuse=True)
		style_B_re, content_A_re, content_A_re_layers = self.Encoder(sty_B_con_A, reuse=True)

		# decode again for cycle loss
		cycle_A = self.Decoder(style_A, content_A_re, content_A_re_layers, reuse=True)
		cycle_B = self.Decoder(style_B, content_B_re, content_B_re_layers, reuse=True)

		real_A_logit, real_B_logit = self.Discriminator_real(self.domain_A, self.domain_B)
		fake_A_logit, fake_B_logit = self.Discriminator_fake(sty_A_con_B, sty_B_con_A)

		# define loss
		G_loss_A = generator_loss(fake_A_logit)
		G_loss_B = generator_loss(fake_B_logit)

		recon_loss_A = L1_loss(sty_A_con_A, self.domain_A) # reconstruction
		recon_loss_B = L1_loss(sty_B_con_B, self.domain_B) # reconstruction

		recon_style_loss_A = L1_loss(style_A_re, style_A)
		recon_style_loss_B = L1_loss(style_B_re, style_B)

		recon_content_loss_A = L1_loss(content_A_re, content_A)
		recon_content_loss_B = L1_loss(content_B_re, content_B)

		cycle_A_loss = L1_loss(cycle_A, self.domain_A)
		cycle_B_loss = L1_loss(cycle_B, self.domain_B)

		D_loss_A = discriminator_loss(real_A_logit, fake_A_logit)
		D_loss_B = discriminator_loss(real_B_logit, fake_B_logit)

		self.style_encoder_loss = recon_style_loss_A + recon_style_loss_B
		self.content_encoder_loss = recon_content_loss_A + recon_content_loss_B

		self.Generator_loss = self.gan_w * (G_loss_A+G_loss_B) + \
						self.rec_w * (recon_loss_A + recon_loss_B) + \
						self.sty_w * (recon_style_loss_A + recon_style_loss_B) + \
						self.con_w * (recon_content_loss_A + recon_content_loss_B) + \
						self.cyc_w * (cycle_A_loss + cycle_B_loss)

		self.Discriminator_loss = D_loss_A + D_loss_B

		''' Training Operations '''
		all_tf_vars = tf.trainable_variables()
		style_vars = [var for var in all_tf_vars if 'style' in var.name]
		content_vars = [var for var in all_tf_vars if 'content' in var.name]
		G_vars = [var for var in all_tf_vars if 'decoder' in var.name or 'encoder' in var.name]
		D_vars = [var for var in all_tf_vars if 'discriminator' in var.name]

		self.style_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, \
			beta2=0.999).minimize(self.style_encoder_loss, var_list=style_vars)
		self.content_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, \
			beta2=0.999).minimize(self.content_encoder_loss, var_list=content_vars)
		
		self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, \
			beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
		self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, \
			beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

		''' Summary '''
		self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
		self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
		self.style_loss = tf.summary.scalar("Style_loss", self.style_encoder_loss)
		self.content_loss = tf.summary.scalar("Content_loss", self.content_encoder_loss)
		self.D_A_loss = tf.summary.scalar("D_A_loss", D_loss_A)
		self.D_B_loss = tf.summary.scalar("D_B_loss", D_loss_B)

		self.G_loss = tf.summary.merge([self.all_G_loss, self.style_loss, self.content_loss])
		self.D_loss = tf.summary.merge([self.all_D_loss, self.D_A_loss, self.D_B_loss])

		''' Images '''

		self.fake_A = sty_A_con_B
		self.fake_B = sty_B_con_A
		self.real_A = self.domain_A
		self.real_B = self.domain_B
		self.reco_A = sty_A_con_A
		self.reco_B = sty_B_con_B
		self.cycl_A = cycle_A
		self.cycl_B = cycle_B

		''' Test '''
		self.test_A = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_c], name='test_A')
		self.test_B = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_c], name='test_B')

		test_style_A, test_content_A, test_content_A_layers = self.Encoder(self.test_A, reuse=True)
		test_style_B, test_content_B, test_content_B_layers = self.Encoder(self.test_B, reuse=True)

		self.test_sty_A_con_B = self.Decoder(test_style_A, test_content_B, test_content_B_layers, reuse=True)
		self.test_sty_B_con_A = self.Decoder(test_style_B, test_content_A, test_content_A_layers, reuse=True)

		test_style_A_re, test_content_B_re, test_content_B_re_layers = self.Encoder(self.test_sty_A_con_B, reuse=True)
		test_style_B_re, test_content_A_re, test_content_A_re_layers = self.Encoder(self.test_sty_B_con_A, reuse=True)
		self.test_recon_A = self.Decoder(test_style_A, test_content_A, test_content_A_re_layers, reuse=True)
		self.test_recon_B = self.Decoder(test_style_B, test_content_B, test_content_B_re_layers, reuse=True)

		self.test_cycle_A = self.Decoder(test_style_A, test_content_A_re, test_content_A_re_layers, reuse=True)
		self.test_cycle_B = self.Decoder(test_style_B, test_content_B_re, test_content_B_re_layers, reuse=True)
		
	##################################################################################
	# Operations
	##################################################################################

	def train(self):
		# initialize all variables
		tf.global_variables_initializer().run()

		# saver to save model
		self.saver = tf.train.Saver()

		# summary writer
		self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

		# load old checkpoints
		loadable, checkpoint_counter = self.load(self.checkpoint_dir)
		if(loadable):
			start_epoch = (int)(checkpoint_counter/self.iteration)
			start_batch_id = checkpoint_counter - start_epoch * self.iteration
			counter = checkpoint_counter
			print(' [*]----- Succefully Load Checkpoint -----[*] ')
		else:
			start_epoch = 0
			start_batch_id = 0
			counter = 0
			print(' [!]----- Fail to Load Checkpoint -----[!] ')

		# start loopingfor epochs
		start_time = time.time()

		index_path = os.path.join(self.sample_dir, 'index.html')
		index = open(index_path, 'a')
		index.write("<html><body><table><tr>")
		index.write("<th>image_A</th> <th>image_B</th><th>styA_conB</th><th>styB_conA</th>\
			<th>recon_A</th><th>recon_B</th><th>cycle_A</th><th>cycle_B</th></tr>")
		index.close
		for epoch in range(start_epoch, self.epoch+1):
			lr = self.init_lr * pow(0.5, epoch)

			# defaul interation = 100000
			for idx in range(start_batch_id, self.iteration):

				# Update G
				batch_A_images, batch_B_images, fake_A, fake_B, reco_A, reco_B, cycl_A, cycle_B, _, g_loss, summary_str\
					= self.sess.run([
						self.real_A, 
						self.real_B, 
						self.fake_A, 
						self.fake_B,
						self.reco_A, 
						self.reco_B, 
						self.cycl_A, 
						self.cycl_B, 
						self.G_optim, 
						self.Generator_loss, 
						self.G_loss], feed_dict={self.lr:lr})
				self.writer.add_summary(summary_str, counter)

				# update style and content encoder

				# Update D
				_, d_loss, summary_str = self.sess.run([
					self.D_optim, 
					self.Discriminator_loss, 
					self.D_loss], feed_dict={self.lr:lr})
				self.writer.add_summary(summary_str, counter)

				counter += 1
				# log out training status now
				print('Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.6f, g_loss: %.6f'\
					%(epoch, idx, self.iteration, time.time()-start_time, d_loss, g_loss))

				if(np.mod(idx+1, self.print_freq)==0):

					index = open(index_path, 'a')
					save_images(batch_A_images, self.batch_size,\
						'./{}/real_A_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
					save_images(batch_B_images, self.batch_size,\
						'./{}/real_B_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))

					save_images(fake_A, self.batch_size,\
						'./{}/fake_A_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
					save_images(fake_B, self.batch_size,\
						'./{}/fake_B_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
					save_images(reco_A, self.batch_size,\
						'./{}/reco_A_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
					save_images(reco_B, self.batch_size,\
						'./{}/reco_B_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
					save_images(cycl_A, self.batch_size,\
						'./{}/cycl_A_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))
					save_images(cycle_B, self.batch_size,\
						'./{}/cycl_B_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1))

					index.write("<tr>")
					index.write("<td><img src=real_A_{:02d}_{:06d}.jpg width={:d} height={:d}></td>".format(epoch, idx+1 , 150, 150))
					index.write("<td><img src=real_B_{:02d}_{:06d}.jpg width={:d} height={:d}></td>".format(epoch, idx+1 , 150, 150))
					
					index.write("<td><img src=fake_A_{:02d}_{:06d}.jpg width={:d} height={:d}></td>".format( epoch, idx+1 , 150, 150))
					index.write("<td><img src=fake_B_{:02d}_{:06d}.jpg width={:d} height={:d}></td>".format( epoch, idx+1 , 150, 150))
					index.write("<td><img src=reco_A_{:02d}_{:06d}.jpg width={:d} height={:d}></td>".format( epoch, idx+1 , 150, 150))
					index.write("<td><img src=reco_B_{:02d}_{:06d}.jpg width={:d} height={:d}></td>".format( epoch, idx+1 , 150, 150))
					index.write("<td><img src=cycl_A_{:02d}_{:06d}.jpg width={:d} height={:d}></td>".format( epoch, idx+1 , 150, 150))
					index.write("<td><img src=cycl_B_{:02d}_{:06d}.jpg width={:d} height={:d}></td>".format( epoch, idx+1 , 150, 150))
					index.write("</tr>")
					index.close()

				if(np.mod(idx+1, self.save_freq)==0):
					self.save(counter)
			start_batch_id = 0
			self.save(counter)

	def test(self):
		tf.global_variables_initializer().run()
		test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
		test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		self.result_dir = os.path.join(self.result_dir, self.model_dir)
		check_folder(self.result_dir)

		if could_load :
			print(" [*] Load SUCCESS")
		else :
			print(" [!] Load failed...")

		index_path = os.path.join(self.result_dir, 'index.html')
		index = open(index_path, 'w')
		index.write("<html><body><table><tr>")
		index.write("<th>image_A</th> <th>image_B</th><th>styA_conB</th><th>styB_conA</th>\
			<th>recon_A</th><th>recon_B</th><th>cycle_A</th><th>cycle_B</th></tr>")

		list_len = min(len(test_A_files), len(test_B_files))
		for idx in range(list_len):
			image_A = np.asarray(load_test_data(test_A_files[idx], size_h=self.img_h, size_w=self.img_w))
			image_B = np.asarray(load_test_data(test_B_files[idx], size_h=self.img_h, size_w=self.img_w))
			base_path = os.path.basename(test_A_files[idx]).split(".")[0]
			image_path = os.path.join(self.result_dir, base_path)

			test_sty_A_con_B, test_sty_B_con_A, test_recon_A, test_recon_B, test_cycle_A, test_cycle_B = \
			self.sess.run([self.test_sty_A_con_B, self.test_sty_B_con_A, self.test_recon_A, self.test_recon_B, self.test_cycle_A, self.test_cycle_B],\
				feed_dict={self.test_A : image_A, self.test_B : image_B})
			save_images(image_A, 1, image_path+'image_A.jpg')
			save_images(image_B, 1, image_path+'image_B.jpg')
			save_images(test_sty_A_con_B, 1, image_path+'test_sty_A_con_B.jpg')
			save_images(test_sty_B_con_A, 1, image_path+'test_sty_B_con_A.jpg')
			save_images(test_recon_A, 1, image_path+'test_recon_A.jpg')
			save_images(test_recon_B, 1, image_path+'test_recon_B.jpg')
			save_images(test_cycle_A, 1, image_path+'test_cycle_A.jpg')
			save_images(test_cycle_B, 1, image_path+'test_cycle_B.jpg')
			index.write("<tr>")
			index.write("<td><img src='%s' width='%d' height='%d'></td>" % (base_path+'image_A.jpg' , 150, 150))
			index.write("<td><img src='%s' width='%d' height='%d'></td>" % (base_path+'image_B.jpg' , 150, 150))
			
			index.write("<td><img src='%s' width='%d' height='%d'></td>" % ( base_path+'test_sty_A_con_B.jpg' , 150, 150))
			index.write("<td><img src='%s' width='%d' height='%d'></td>" % ( base_path+'test_sty_B_con_A.jpg' , 150, 150))
			index.write("<td><img src='%s' width='%d' height='%d'></td>" % ( base_path+'test_recon_A.jpg' , 150, 150))
			index.write("<td><img src='%s' width='%d' height='%d'></td>" % ( base_path+'test_recon_B.jpg' , 150, 150))
			index.write("<td><img src='%s' width='%d' height='%d'></td>" % ( base_path+'test_cycle_A.jpg' , 150, 150))
			index.write("<td><img src='%s' width='%d' height='%d'></td>" % ( base_path+'test_cycle_B.jpg' , 150, 150))
			index.write("</tr>")

		index.close()

	@property
	def model_dir(self):
		return "{}_{}".format(self.model_name, self.dataset_name)


	def save(self, step):
		checkpoint_dir = os.path.join(self.checkpoint_dir)	
		check_folder(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading old checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0



