import os
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image

class txtfile_classification(Dataset):

    # clip duration = 16, i.e, for each time 16 frames are considered together
    def __init__(self, base, root, transform=None,):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        self.base_path = base
        self.nSamples  = len(self.lines)
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        im_split = imgpath.split(',')

        path_tmp = self.base_path + im_split[0]

        frame = Image.open(path_tmp).convert('RGB')
        frame = self.transform(frame)

        yoga6=np.array(im_split[1], dtype=np.int64)
        yoga6 = torch.from_numpy(yoga6)

        yoga20=np.array(im_split[2], dtype=np.int64)
        yoga20 = torch.from_numpy(yoga20)

        yoga82=np.array(im_split[3], dtype=np.int64)
        yoga82 = torch.from_numpy(yoga82)

        return (frame, yoga6, yoga20, yoga82)



class txtfile_classification_cscl(Dataset):

    def __init__(self, base, root, transform=None,):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        self.base_path = base
        self.nSamples  = len(self.lines)
        self.transform = transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        im_split = imgpath.split(',')

        path_tmp = self.base_path + im_split[0]

        frame = Image.open(path_tmp).convert('RGB')
        #frame = self.transform(frame)
        #frame=TwoCropTransform(frame)
        frame = [self.transform(frame), self.transform(frame)]
        #print("#################################1")
        #print(frame.shape)

        yoga6=np.array(im_split[1], dtype=np.int64)
        yoga6 = torch.from_numpy(yoga6)

        yoga20=np.array(im_split[2], dtype=np.int64)
        yoga20 = torch.from_numpy(yoga20)

        yoga82=np.array(im_split[3], dtype=np.int64)
        yoga82 = torch.from_numpy(yoga82)

        return (frame, yoga6, yoga20, yoga82)

