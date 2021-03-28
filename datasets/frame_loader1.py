import cv2, json
import os, random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy import ndimage
from torch.utils.data import DataLoader

def random_crop(img, size, image_shape):
    h, w = image_shape
    th, tw = size
    h1 = torch.randint(0, h - th + 1, (1,)).item() if (h - th)>0 else 0
    w1 = torch.randint(0, w - tw + 1, (1,)).item() if (w - tw)>0 else 0
    return img[h1:(h1 + th), w1:(w1 + tw), :]


class YoutubeVOS(Dataset):
    def __init__(self,
                 mode,
                 json_path,
                 im_path,
                 ann_path,
                 transform=None,
                 hflip=False,
                 max_len=5):

        self.transform = transform
        self.mode = mode
        self.hflip = hflip
        self.max_len = max_len

        with open(json_path, 'r') as f:
            data = f.read()

        self.obj = json.loads(data)

        self.im_path = im_path
        self.ann_path = ann_path

        # list of (sequence, obj_id)
        if self.mode is 'train':
            seqs = list(self.obj['videos'].keys())
            self.sequences = []

            for seq in seqs:
                categories = list(self.obj['videos'][seq]['objects'].keys())
                for cat in categories:
                    self.sequences.append((seq, cat))

        elif self.mode is 'test':
            self.sequences = list(self.obj['videos'].keys())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """ returns a list of frames available under the object key of a sequence.
        for test mode returns a dict of all frames for each object """

        if self.mode is 'train':
            this_seq, obj = self.sequences[idx]
            seq_dict = {'image': [], 'frames':[]}
            obj_frames = self.obj['videos'][this_seq]['objects'][obj]['frames']
            seq_len = len(obj_frames)

            starting_idx = obj_frames.index(random.choice(obj_frames[:-self.max_len])) if seq_len > self.max_len else 0
            selected_frames = obj_frames[starting_idx: (self.max_len + starting_idx)]

            self.make_img_gt_pair_train(selected_frames, this_seq, seq_dict)

        elif self.mode is 'test':
            seq_dict = {'seq_name': self.sequences[idx]}

            categories = list(self.obj['videos'][self.sequences[idx]]['objects'].keys())
            img_path = self.im_path + self.sequences[idx] + '/'

            for cat in categories:
                seq_dict[cat] = {'image': [], 'name': []}

                selected_frames_total = self.obj['videos'][self.sequences[idx]]['objects'][cat]['frames']
                img_0, first_mask = self.make_img_gt_pair_test(self.sequences[idx], selected_frames_total[0], int(cat))

                seq_dict[cat]['image'].append(img_0)
                seq_dict[cat]['name'].append(selected_frames_total[0])
                seq_dict[cat]['first_mask'] = first_mask

                for f_name in selected_frames_total[1:]:
                    img = (cv2.imread(img_path + f_name + '.jpg'))
                    img = torch.from_numpy(img)
                    seq_dict[cat]['image'].append(img)
                    seq_dict[cat]['name'].append(f_name)

        return seq_dict

    def make_img_gt_pair_train(self, frame_list, seq_name, seq_dict):
        """ returns pair of rgb and binary mask, where the mask is available
            data aug ref: https://github.com/linjieyangsc/video_seg/blob/master/dataset_davis.py
        """
        img_path = os.path.join(self.im_path, seq_name)
        if self.hflip:
            flip = True if random.random() > 0.5 else False
        else:
            flip = False
        
        if flip:
            frame_list = frame_list[::-1]

        for frame in frame_list:
            img = (cv2.imread(os.path.join(img_path, frame + '.jpg')))
            img = random_crop(img, (256,256), img.shape[:2])
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            seq_dict['image'].append(img)
            seq_dict['frames'].append(frame_list)

    def make_img_gt_pair_test(self, seq_name, frame, obj):
        img_path = os.path.join(self.im_path, seq_name)
        label_path = os.path.join(self.ann_path, seq_name)
        img = (cv2.imread(os.path.join(img_path, frame + '.jpg')))
        img = torch.from_numpy(img)
        return img

if __name__ == "__main__":
    tr_conf = {
        'num_classes':42,
        'border_pixels':20,
        'bin_size':1,
        'n_epoch': 400,
        'b_s': 4,
        'n_workers': 4,
        'optimizer': 'Adam',
        'reduction': 'Mean',
        'lr': 1e-5,
        'starting_epoch':0,
        #/ds/videos/YoutubeVOS2018/train
        'meta_train_path': '/netscratch/kadur/lable_propagation/augmentation/YoutubeVOS2018/train/meta_all.json',
        'im_train_path': '/netscratch/kadur/lable_propagation/augmentation/YoutubeVOS2018/train/JPEGImages/',
        'ann_train_path': '/netscratch/kadur/lable_propagation/augmentation/YoutubeVOS2018/train/Annotations/',
        'affine_info': {
            'angle': range(-20, 20),
            'translation': range(-10, 10),
            'scale': range(75, 125),
            'shear': range(-10, 10)},
        'hflip': True,
        'lambda1': 0.6,
        'lambda2': 0.2,
        'lambda3': 0.2
    }

    im_res = [256, 448]

    train_set = YoutubeVOS(mode='train',
                            json_path=tr_conf['meta_train_path'],
                            im_path=tr_conf['im_train_path'],
                            ann_path=tr_conf['ann_train_path'],
                            transform=None,
                            hflip=tr_conf['hflip'],
                            max_len = 5)
    train_loader = DataLoader(train_set, batch_size=tr_conf['b_s'], num_workers=1,
                                      shuffle=True, pin_memory=True)
    for i,sample in enumerate(train_loader):
        print(sample['frames'][1])
        print(len(sample['image']))
        break
