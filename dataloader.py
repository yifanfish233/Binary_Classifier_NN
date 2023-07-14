from random import random

import cv2
import os
import torch
from torch.utils.data import Dataset

from glob import glob
from random import shuffle
# seed = 0
# rand = random.seed(seed)
class CelebA(Dataset):
    def __init__(self, image_dir,usage, shuffle):
        self.paths = []
        self.usage = usage
        self.shuffle = shuffle
        #
        if(shuffle =="yes"):
            if usage == "test":
                # self.paths=glob(os.path.join(image_dir, '*.png'))[:int(202599 * 0.3)]
                self.paths=glob(os.path.join(image_dir, '*.png'))[:int(500 * 0.3)]
            elif usage == "train":
                self.paths=glob(os.path.join(image_dir, '*.png'))[int(500 * 0.3):]
                # self.paths=glob(os.path.join(image_dir, '*.png'))[int(202599 * 0.3):]
        else:
            if usage == "train":
                self.paths=glob(os.path.join(image_dir, '*.png'))[:int(500*0.7)]
                # self.paths=glob(os.path.join(image_dir, '*.png'))[:int(202599*0.7)]
            elif usage == "test":
                self.paths=glob(os.path.join(image_dir, '*.png'))[int(500*0.7):]
                # self.paths=glob(os.path.join(image_dir, '*.png'))[int(202599*0.7):]
        if usage == "all":
            self.paths = glob(os.path.join(image_dir, '*.png'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #resize
        im = cv2.resize(im, (160, 192))
        im = im.astype('float32') / 255
        # im = torch.from_numpy(im).permute(2, 0, 1)
        im = im.transpose(2, 0, 1)
        torch_im = torch.from_numpy(im).float()
        name = os.path.basename(path)[:-4]
        label = int(name.endswith('m'))
        return torch_im, label, os.path.basename(path)
