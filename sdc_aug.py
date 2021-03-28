import os 
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
import shutil
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.sdc_net2d import *

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to trained video reconstruction checkpoint')
parser.add_argument('--flownet2_checkpoint', default='', type=str, metavar='PATH', help='path to flownet-2 best checkpoint')
parser.add_argument('--source_dir', default='', type=str, help='directory for data (default: Cityscapes root directory)')
parser.add_argument('--target_dir', default='', type=str, help='directory to save augmented data')
parser.add_argument('--sequence_length', default=2, type=int, metavar="SEQUENCE_LENGTH",
                    help='number of interpolated frames (default : 2)')
parser.add_argument("--rgb_max", type=float, default = 255.)
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--propagate', type=int, default=3, help='propagate how many steps')
parser.add_argument('--vis', action='store_true', default=False, help='augment color encoded segmentation map')
parser.add_argument('--scene', default=None, help='augment a single scene, pass the scene id here.')
parser.add_argument('--size', default=[448, 256], nargs='+', help='dataset images will to resized to this size')
parser.add_argument('--stride', type=int, default=64, help='The factor for which the image sizes should be evenly divisible')

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
                        64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]

def get_model():
	model = SDCNet2DRecon(args)
	checkpoint = torch.load(args.pretrained)
	args.start_epoch = 0 if 'epoch' not in checkpoint else checkpoint['epoch']
	state_dict = checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict']
	model.load_state_dict(state_dict, strict=False)
	logging.info("Loaded checkpoint '{}' (at epoch {})".format(args.pretrained, args.start_epoch))
	return model

def process_image(img_file):
	img_rgb = cv2.imread(img_file)
	height, width = img_rgb.shape[:2]
	if height % args.stride !=0 or width % args.stride != 0:
		img_rgb = cv2.resize(img_rgb, (args.size[0], args.size[1]))
	img_rgb = img_rgb.transpose((2,0,1))
	img_rgb = np.expand_dims(img_rgb, axis=0)
	img_rgb = torch.from_numpy(img_rgb.astype(np.float32))
	return img_rgb

def process_mask(gt_labelid_dir):
	gt_labelid = Image.open(gt_labelid_dir).resize(args.size)
	gt_labelid = np.array(gt_labelid)
	gt_labelid = np.expand_dims(gt_labelid, axis=0)
	gt_labelid = np.expand_dims(gt_labelid, axis=0)
	gt_labelid = torch.from_numpy(gt_labelid.astype(np.float32))
	return gt_labelid

def get_data(img_dir, gt_labelid_dir):
	imgs_rgb = []
	for img_file in img_dir:
		imgs_rgb.append(process_image(img_file))
	gt_labelid = process_mask(gt_labelid_dir)
	return imgs_rgb, gt_labelid

def generate_missing_frames(model, mask_prefix, rgb_prefix, sequence_prefix, split, scene):
	scene_dir = os.path.join(args.source_dir, split, rgb_prefix, scene)
	frames = os.listdir(scene_dir)
	frames.sort()

	if not os.path.exists(os.path.join(args.target_dir, split, rgb_prefix, scene, frames[0])):
		frame = frames[1]
		reverse = True
		for propagate in range(4):
			augment_data(model, mask_prefix, rgb_prefix, sequence_prefix, split, scene, 'rgb_image', reverse, frame, propagate)
			augment_data(model, mask_prefix, rgb_prefix, sequence_prefix, split, scene, 'labelid', reverse, frame, propagate)
		# copy first image and mask to target directory
		img_rgb = process_image(os.path.join(args.source_dir, split, rgb_prefix, scene, frames[0]))
		mask = process_mask(os.path.join(args.source_dir, split, mask_prefix, scene, frames[0].split(".")[0]+".png"))
		cv2.imwrite(os.path.join(args.target_dir, split, rgb_prefix, scene, frames[0]), img_rgb.data.cpu().numpy().squeeze().transpose(1,2,0).astype(np.uint8))
		res_img = Image.fromarray(mask.data.cpu().numpy().squeeze().astype(np.uint8), mode='P')
		res_img.putpalette(palette)
		res_img.save(os.path.join(args.target_dir, split, mask_prefix, scene, frames[0].split(".")[0]+".png"))

	if not os.path.exists(os.path.join(args.target_dir, split, rgb_prefix, scene, frames[-1])):
		frame = frames[-2]
		reverse = False
		for propagate in range(4):
			augment_data(model, mask_prefix, rgb_prefix, sequence_prefix, split, scene, 'rgb_image', reverse, frame, propagate)
			augment_data(model, mask_prefix, rgb_prefix, sequence_prefix, split, scene, 'labelid', reverse, frame, propagate)
		# copy first image and mask to target directory
		img_rgb = process_image(os.path.join(args.source_dir, split, rgb_prefix, scene, frames[-1]))
		mask = process_mask(os.path.join(args.source_dir, split, mask_prefix, scene, frames[-1].split(".")[0]+".png"))
		cv2.imwrite(os.path.join(args.target_dir, split, rgb_prefix, scene, frames[-1]), img_rgb.data.cpu().numpy().squeeze().transpose(1,2,0).astype(np.uint8))
		res_img = Image.fromarray(mask.data.cpu().numpy().squeeze().astype(np.uint8), mode='P')
		res_img.putpalette(palette)
		res_img.save(os.path.join(args.target_dir, split, mask_prefix, scene, frames[-1].split(".")[0]+".png"))

def augment_data(model, mask_prefix, rgb_prefix, sequence_prefix, split, scene, mode, reverse, frame, propagate):
	seq_info = frame.split(".")[0]
	# get rgb images
	source_imgs = list()

	if not reverse:
		curr_frame = "%05d" % (int(seq_info) + propagate)
		# get previous images
		for i in range(args.sequence_length-1):
			seq_id = "%05d" % (int(curr_frame) - i)
			img_name = seq_id+'.jpg'
			#source_imgs.append(os.path.join(args.source_dir, sequence_prefix, rgb_prefix, scene, img_name))
			if os.path.isfile(os.path.join(args.target_dir, split, rgb_prefix, scene, img_name)):
				source_imgs.append(os.path.join(args.target_dir, split, rgb_prefix, scene, img_name))
			else:
				source_imgs.append(os.path.join(args.source_dir, sequence_prefix, rgb_prefix, scene, img_name))
			source_imgs = list(reversed(source_imgs))

		# append target image
		target_frame = "%05d" % (int(curr_frame) + 1) + ".jpg"
		target_img = os.path.join(args.source_dir, sequence_prefix, rgb_prefix, scene, target_frame)
		source_imgs.append(target_img)
		# get the mask image
		mask_name =  "%05d" % (int(curr_frame)) + '.png'
		if propagate == 0:
			mask_image = os.path.join(args.source_dir, split, mask_prefix, scene, mask_name)
		else:
			mask_image = os.path.join(args.target_dir, split, mask_prefix, scene, mask_name)
	else:
		curr_frame = "%05d" % (int(seq_info) - propagate)
		# get next images if reverse is true
		for i in range(args.sequence_length-1):
			seq_id = "%05d" % (int(curr_frame) + i)
			img_name = seq_id+".jpg"
			#source_imgs.append(os.path.join(args.source_dir, sequence_prefix, rgb_prefix, scene, img_name))
			if os.path.isfile(os.path.join(args.target_dir, split, rgb_prefix, scene, img_name)):
				source_imgs.append(os.path.join(args.target_dir, split, rgb_prefix, scene, img_name))
			else:
				source_imgs.append(os.path.join(args.source_dir, sequence_prefix, rgb_prefix, scene, img_name))
			source_imgs = list(reversed(source_imgs))
		curr_img_name = curr_frame + '.jpg'
		# append target image
		target_frame = "%05d" % (int(curr_frame) - 1) + ".jpg"
		target_img = os.path.join(args.source_dir, sequence_prefix, rgb_prefix, scene, target_frame)
		source_imgs.append(target_img)
		# get the mask image
		mask_name =  "%05d" % (int(curr_frame)) + '.png'
		if propagate == 0:
			mask_image = os.path.join(args.source_dir, split, mask_prefix, scene, mask_name)
		else:
			mask_image = os.path.join(args.target_dir, split, mask_prefix, scene, mask_name)
	
	imgs_rgb, gt_label_id = get_data(source_imgs, mask_image)
	
	imgs_rgb = [Variable(img_rgb).contiguous().cuda() for img_rgb in imgs_rgb]
	gt_label_id = Variable(gt_label_id).contiguous().cuda()
	input_dict = {}
	input_dict['image'] = imgs_rgb
	
	if mode == "rgb_image":
		_, pred_rgb, _ = model(input_dict)
		pred_rgb_img = ( pred_rgb.data.cpu().numpy().squeeze().transpose(1,2,0) ).astype(np.uint8)
		
		if not os.path.exists(os.path.join(args.target_dir, split, rgb_prefix, scene)):
			os.makedirs(os.path.join(args.target_dir, split, rgb_prefix, scene))
		
		target_name_prev = "%05d" % (int(curr_frame) - 1)+'.jpg'
		target_name_next = "%05d" % (int(curr_frame) + 1)+'.jpg'
		
		target_im_prev = os.path.join(args.target_dir, split, rgb_prefix, scene, target_name_prev)
		target_im_curr = os.path.join(args.target_dir, split, rgb_prefix, scene, curr_frame+'.jpg')
		target_im_next = os.path.join(args.target_dir, split, rgb_prefix, scene, target_name_next)
		
		if propagate == 0:
			cv2.imwrite(target_im_curr, (imgs_rgb[-2].data.cpu().numpy().squeeze().transpose(1,2,0)).astype(np.uint8))
		
		if not reverse:
			cv2.imwrite(target_im_next, pred_rgb_img)
		else:
			cv2.imwrite(target_im_prev, pred_rgb_img)
	
	elif mode == "labelid":
		_, pred_labelid, _ = model(input_dict, label_image=gt_label_id)
		pred_labelid_img = pred_labelid.data.cpu().numpy().squeeze().astype(np.uint8)
		
		if not os.path.exists(os.path.join(args.target_dir, split, mask_prefix, scene)):
			os.makedirs(os.path.join(args.target_dir, split, mask_prefix, scene))
		
		target_labelid = os.path.join(args.target_dir, split, mask_prefix, scene, mask_name)
		
		if propagate == 0:
			res_img = Image.fromarray(gt_label_id.data.cpu().numpy().squeeze().astype(np.uint8), mode='P')
			res_img.putpalette(palette)
			res_img.save(target_labelid)
		
		if not reverse:
			labelid_gt_name = "%05d" % (int(curr_frame) + 1) + ".png"
			target_labelid = os.path.join(args.target_dir, split, mask_prefix, scene, labelid_gt_name)
			res_img = Image.fromarray(pred_labelid_img, mode='P')
			res_img.putpalette(palette)
			res_img.save(target_labelid)
		else:
			labelid_gt_name = "%05d" % (int(curr_frame) - 1) + ".png"
			target_labelid = os.path.join(args.target_dir, split, mask_prefix, scene, labelid_gt_name)
			res_img = Image.fromarray(pred_labelid_img, mode='P')
			res_img.putpalette(palette)
			res_img.save(target_labelid)
	else:
		logging.info("Mode %s is not supported." % (mode))
		sys.exit()
		

def augmentation(model, mask_prefix, rgb_prefix, sequence_prefix, split, scene, mode, reverse, propagate):
	scene_dir = os.path.join(args.source_dir, split, rgb_prefix, scene)
	frames = os.listdir(scene_dir)
	frames.sort()

	for frame in frames[1:-1]:
		augment_data(model, mask_prefix, rgb_prefix, sequence_prefix, split, scene, mode, reverse, frame, propagate)

if __name__ == '__main__':
	global args
	args = parser.parse_args()
	if not os.path.exists(args.target_dir):
		os.mkdir(args.target_dir)
	logging.basicConfig(filename=os.path.join(args.target_dir, 'stdout.log'), level=logging.DEBUG)

	# Load pre-trained video reconstruction model
	net = get_model()
	net.eval()
	net = net.cuda()

	# Config paths
	if not os.path.exists(args.target_dir):
		os.makedirs(args.target_dir)

	mask_prefix = "Annotations"
	rgb_prefix = "JPEGImages"
	sequence_prefix = "train_all_frames"
	split = "train"

	if not args.scene:
		sequences = os.listdir(os.path.join(args.source_dir, split, rgb_prefix))
		for scene in sequences:
			# Generate augmented dataset
			for i in range(0, args.propagate):
				# create +-n data
				augmentation(net, mask_prefix, rgb_prefix, sequence_prefix, split, scene, 'rgb_image', False, i)
				augmentation(net, mask_prefix, rgb_prefix, sequence_prefix, split, scene, 'rgb_image', True, i)
				augmentation(net, mask_prefix, rgb_prefix, sequence_prefix, split, scene, 'labelid', False, i)
				augmentation(net, mask_prefix, rgb_prefix, sequence_prefix, split, scene, 'labelid', True, i)

			try:
				generate_missing_frames(net, mask_prefix, rgb_prefix, sequence_prefix, split, scene)
			except BaseException as exp:
				print(f'Skipping : {exp}')
				pass
	else:
		# Generate augmented dataset
		for i in range(0, args.propagate):
			# create +-n data
			#augmentation(net, mask_prefix, rgb_prefix, sequence_prefix, split, args.scene, 'rgb_image', False, i)
			augmentation(net, mask_prefix, rgb_prefix, sequence_prefix, split, args.scene, 'rgb_image', True, i)
			#augmentation(net, mask_prefix, rgb_prefix, sequence_prefix, split, args.scene, 'labelid', False, i)
			augmentation(net, mask_prefix, rgb_prefix, sequence_prefix, split, args.scene, 'labelid', True, i)

		try:
			generate_missing_frames(net, mask_prefix, rgb_prefix, sequence_prefix, split, args.scene)
		except BaseException as exp:
			print(f'Skipping : {exp}')
			pass
		
