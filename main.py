from ZSIT_2way_m_2 import ZSIT
from utils import *
import argparse
import os

def parse_args():
	desc = "Tensorflow implementation of ZSIT"
	parser = argparse.ArgumentParser()

	parser.add_argument('--phase', type=str, default='train', help='train or test')
	parser.add_argument('--exp_name', type=str, default='ZSIT_train', help='folder to save all stuff')

	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
		help='Directory name to save the checkpoints')
	parser.add_argument('--result_dir', type=str, default='results',
		help='Directory name to save the generated images')
	parser.add_argument('--log_dir', type=str, default='logs',
		help='Directory name to save training logs')
	parser.add_argument('--sample_dir', type=str, default='samples',
		help='Directory name to save the samples on training')

	parser.add_argument('--datasets', type=list, default=['coco', 'font'], help='dataset_name')
	parser.add_argument('--augment_flag', type=bool, default=True, help='Image augmentation use or not')
	parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
	parser.add_argument('--iteration', type=int, default=1000, help='The number of training iterations')
	parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
	parser.add_argument('--print_freq', type=int, default=50, help='The number of image_print_freq')
	parser.add_argument('--save_freq', type=int, default=50, help='The number of ckpt_save_freq')
	parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')

	parser.add_argument('--img_h', type=int, default=256, help='The size of image hegiht')
	parser.add_argument('--img_w', type=int, default=256, help='The size of image width')
	parser.add_argument('--img_c', type=int, default=3, help='The size of image channel')

	parser.add_argument('--gan_w', type=float, default=1.0, help='weight of adversarial loss')
	parser.add_argument('--con_w', type=float, default=3.0, help='weight of content reconstruction loss')
	parser.add_argument('--sty_w', type=float, default=1.0, help='weight of style reconstruction loss')
	parser.add_argument('--rec_w', type=float, default=1.0, help='weight of image reconstruction loss')
	parser.add_argument('--per_w', type=float, default=0.0, help='weight of perceptual loss')

	parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
	
	parser.add_argument('--n_cont_res', type=int, default=4, help='number of residual blocks in content encoder')
	parser.add_argument('--n_cont_downsample', type=int, default=2, help='number of residual blocks in content encoder')

	parser.add_argument('--n_style_res', type=int, default=4, help='number of residual blocks in style encoder')
	parser.add_argument('--n_style_downsample', type=int, default=2, help='number of residual blocks in style encoder')

	parser.add_argument('--n_upsample', type=int, default=2, help='number of residual blocks in decoder')
	parser.add_argument('--useUNet', type=bool, default=True, help='use UNet or not')
	 
	parser.add_argument('--n_scale', type=int, default=3, help='number of scales of discriminator')
	parser.add_argument('--n_dis', type=int, default=4, help='number of discriminator layer')
	parser.add_argument('--sn', type=bool, default=True, help='use spetral norm or not')
	
	args = parser.parse_args()
	args.checkpoint_dir = os.path.join(args.exp_name, args.checkpoint_dir)
	args.result_dir = os.path.join(args.exp_name, args.result_dir)
	args.log_dir = os.path.join(args.exp_name, args.log_dir)
	args.sample_dir = os.path.join(args.exp_name, args.exp_name+'_'+args.sample_dir)

	check_args(args)

	return args

def check_args(args):
	# --exp dir
	check_folder(args.exp_name)

	# --checkpoint_dir
	check_folder(args.checkpoint_dir)

	# --result_dir
	check_folder(args.result_dir)

	# --result_dir
	check_folder(args.log_dir)

	# --sample_dir
	check_folder(args.sample_dir)

def main():
	args = parse_args()

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		ZSIT_model = ZSIT(sess, args)

		ZSIT_model.build_network()
		show_all_variables()

		if args.phase == 'train' :
			# launch the graph in a session
			ZSIT_model.train()
			print(" [*] Training finished!")
		elif args.phase == 'test' :
			ZSIT_model.test()
			print(" [*] Test finished!")


if __name__ == '__main__':
	main()

