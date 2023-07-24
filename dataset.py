# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[len(words)-1])))

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]   # fn 是一个文件夹的地址
        all_img = []
        for i in range(1,9):
            all_img.append(fn + '/' + str(i) + '.png')
        # img_dir = os.listdir(fn)
        # for _ in img_dir:
        #     all_img.append(fn + '/' + _)
        im1 = Image.open(all_img[0]).convert('RGB')
        im1 = self.transform(im1)
        im2 = Image.open(all_img[1]).convert('RGB')
        im2 = self.transform(im2)
        im3 = Image.open(all_img[2]).convert('RGB')
        im3 = self.transform(im3)
        im4 = Image.open(all_img[3]).convert('RGB')
        im4 = self.transform(im4)
        im5 = Image.open(all_img[4]).convert('RGB')
        im5 = self.transform(im5)
        im6 = Image.open(all_img[5]).convert('RGB')
        im6 = self.transform(im6)
        im7 = Image.open(all_img[6]).convert('RGB')
        im7 = self.transform(im7)
        im8 = Image.open(all_img[7]).convert('RGB')
        im8 = self.transform(im8)
        imgs = torch.cat((im1,im2,im3,im4,im5,im6,im7,im8),0)

        return imgs, label

    def __len__(self):
        return len(self.imgs)