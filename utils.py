import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import os
import numpy as np

class ImageData:
	# define input data info
	def __init__(self, img_h=256, img_w=256, channels=3, augment_flag=False, if_style=False):
		self.img_h = img_h
		self.img_w = img_w
		self.channels = channels
		self.augment_flag = augment_flag
		self.if_style = if_style

	# read image file from file path
	def image_processing(self, filename):
		# print(str(filename))
		x = tf.read_file(filename)
		x_decode = tf.image.decode_jpeg(x, channels=self.channels)
		img = tf.image.resize_images(x_decode, [self.img_h, self.img_w])
		# shift value to -1 ~ 1
		img = tf.cast(img, tf.float32)/127.5-1

		if(self.augment_flag):
			augment_size_h = self.img_h + (30 if self.img_h == 256 else 15)
			augment_size_w = self.img_w + (30 if self.img_w == 256 else 15)
			p = random.random()
			if(p>0.2):
				# random crop and flip
				if(self.if_style):
					img = self.augmentation(img, augment_size_h+100, augment_size_w+100)
				else:
					img = self.augmentation(img, augment_size_h, augment_size_w)

		return img

	def augmentation(self, image, aug_img_h, aug_img_w):
		seed = random.randint(0, 2 ** 31 - 1)
		ori_image_shape = tf.shape(image)
		image = tf.image.random_flip_left_right(image, seed=seed)
		image = tf.image.resize_images(image, [aug_img_h, aug_img_w])
		image = tf.random_crop(image, ori_image_shape, seed=seed)
		return image


def load_test_data(image_path, size_h=256, size_w=256):
	img = misc.imread(image_path, mode='RGB')
	img = misc.imresize(img, [size_h, size_w])
	img = np.expand_dims(img, axis=0)
	img = img/127.5 - 1 # -1 ~ 1 

	return img

def save_images(images, batch_size, image_path):
	images = (images+1.)/2 # 0 ~ 1
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((h*batch_size, w, 3))
	for idx, image in enumerate(images):
		img[h*idx:h*(idx+1), 0:w, :] = image
	return misc.imsave(image_path, img)

def show_all_variables():
	model_vars = tf.trainable_variables()
	slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir


