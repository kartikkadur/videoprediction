from __future__ import division
from __future__ import print_function

import os
#import natsort
import numpy as np
import cv2
import argparse
import random
import logging
import torch

from torch.utils import data
try:
    from datasets.dataset_utils import StaticRandomCrop
except:
    from dataset_utils import StaticRandomCrop

class FrameLoader(data.Dataset):
    def __init__(self, args, root, is_training = False, transform=None):

        self.is_training = is_training
        self.transform = transform
        self.chsize = 3

        # carry over command line arguments
        assert args.sequence_length > 1, 'sequence length must be > 1'
        self.sequence_length = args.sequence_length

        assert args.sample_rate > 0, 'sample rate must be > 0'
        self.sample_rate = args.sample_rate
        # samples every args.sample frames
        self.sample = args.sample

        self.crop_size = args.crop_size
        self.start_index = args.start_index
        self.stride = args.stride

        assert (os.path.exists(root))
        if self.is_training:
            self.start_index = 0

        # collect, colors, motion vectors, and depth
        self.ref = self.collect_filelist(root)

        counts = [((len(el) - self.sequence_length) // (self.sample_rate)) for el in self.ref]
        self.total = np.sum(counts)
        self.cum_sum = list(np.cumsum([0] + [el for el in counts]))

    def collect_filelist(self, root):
        include_ext = [".jpg", "jpeg"]
        # collect subfolders, excluding hidden files, but following symlinks
        dirs = [x[0] for x in os.walk(root, followlinks=True) if not x[0].startswith('.')]

        # naturally sort, both dirs and individual images, while skipping hidden files
        dirs = sorted(dirs)
        # collect datasets
        datasets = []
        for fdir in dirs:
            elements = sorted(os.listdir(fdir))
            frames = []
            for index in range(0, len(elements), self.sample):
                el = elements[index]
                if os.path.isfile(os.path.join(fdir, el)) and\
                not el.startswith('.') and\
                any([el.endswith(ext) for ext in include_ext]):
                    frames.append(os.path.join(fdir, el))
            datasets.append(frames)

        return [el for el in datasets if el]

    def __len__(self):
        return self.total#len(self.ref)#

    def __getitem__(self, index):
        # adjust index
        index = len(self) + index if index < 0 else index
        index = index + self.start_index

        dataset_index = np.searchsorted(self.cum_sum, index + 1)
        index = self.sample_rate * (index - self.cum_sum[np.maximum(0, dataset_index - 1)])

        image_list = self.ref[dataset_index - 1]
        input_files = [ image_list[index + offset] for offset in range(self.sequence_length + 1)]
        
        # get the sequence from index
        #image_list = self.ref[index]

        # randomize the frame selection for rach sequence
        #frame_index = random.randint(0, (len(image_list)-(self.sequence_length+1)))
        #input_files = [ image_list[frame_index + offset] for offset in range(self.sequence_length + 1)]

        # reverse image order with p=0.5
        if self.is_training and torch.randint(0, 2, (1,)).item():
            input_files = input_files[::-1]

        # images = [imageio.imread(imfile)[..., :self.chsize] for imfile in input_files]
        images = [cv2.imread(imfile)[..., :self.chsize] for imfile in input_files]
        input_shape = images[0].shape[:2]
        
        if self.is_training:
            cropper = StaticRandomCrop(self.crop_size, input_shape)
            images = list(map(cropper, images))
        else:
            images = [cv2.resize(img, (256, 256)) for img in images]

        # Pad images along height and width to fit them evenly into models.
        input_shape = images[0].shape[:2]
        height, width = input_shape
        if (height % self.stride) != 0:
            padded_height = (height // self.stride + 1) * self.stride
            images = [ np.pad(im, ((0, padded_height - height), (0,0), (0,0)), 'reflect') for im in images]

        if (width % self.stride) != 0:
            padded_width = (width // self.stride + 1) * self.stride
            images = [np.pad(im, ((0, 0), (0, padded_width - width), (0, 0)), 'reflect') for im in images]

        input_images = [torch.from_numpy(im.transpose(2, 0, 1)).float() for im in images]

        output_dict = {
            'image': input_images, 'ishape': input_shape, 'input_files': input_files
        }

        return output_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset class for YoutubeVOS dataset')
    parser.add_argument('--sequence_length', default=5, type=int, metavar="SEQUENCE_LENGTH",
                    help='number of interpolated frames (default : 2)')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256], metavar="CROP_SIZE",
                    help="Spatial dimension to crop training samples for training (default : [448, 448])")
    parser.add_argument('--stride', default=64, type=int,
                    help='The factor for which padded validation image sizes should be evenly divisible. (default: 64)')
    parser.add_argument("--sample_rate", type=int, default=5,
                    help="step size in looping through datasets")
    parser.add_argument("--start_index", type=int, default=0, metavar="START_INDEX",
                    help="Index to start loading input data (default : 0)")
    parser.add_argument("--sample", type=int, default=1, metavar="SAMPLE",
                    help="Sample rate between frames (default : 1)")
    args = parser.parse_args()
    logging.basicConfig(filename='stdout.log', level=logging.DEBUG)
    dataset_root = '/netscratch/kadur/lable_propagation/augmentation/test/train/JPEGImages'
    dataset = FrameLoader(args, dataset_root, True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, drop_last = True)
    
    for i,sample in enumerate(dataloader): 
        logging.info(sample["input_files"])

    logging.info('Number of Training Images: {}:{}'.format(len(dataloader.dataset), len(dataloader)))
