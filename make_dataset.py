import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 
from PIL import Image, ImageFilter
import random
from torch.utils.data import Dataset, DataLoader
import torch
from pylab import *
import constants as ct

TRANSFORM = True
VISUALIZE = 0
VALIDATION_PERCENT = 20
TEST_PERCENT = 20
DATA_PATH = ct.DATA_PATH

def histeq(im,nbr_bins = 256):
    im = np.array(im)
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed= True)
    cdf = imhist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape),cdf

def gen_data():

    if 0:
        delList = os.listdir(ct.TXT_PATH )
        for f in delList:
            filePath = os.path.join( ct.TXT_PATH , f )
            if os.path.isfile(filePath):
                print(filePath)
                os.remove(filePath)
                print (filePath + " was removed!")
               
    path=DATA_PATH
    dirs = os.listdir(path)
    c = 0
    for dir in dirs:
        c += 1
        print(c, dir,' is generating')
        files = os.listdir(os.path.join(path,dir))
        for img in files:
            if 'im1.jpg' in img:
                img1_path = os.path.join(dir, img)
            elif 'im2.jpg' in img :
                img2_path = os.path.join(dir,img)
            elif 'gt.jpg' in img:
                gt_path = os.path.join(dir,img)        
          
        chance = np.random.randint(100)
        if chance<VALIDATION_PERCENT:
            with open(os.path.join(ct.TXT_PATH,'validation.txt'),'a') as f:
                f.write(img1_path+','+img2_path+','+gt_path)
                f.write('\n')
        elif chance<VALIDATION_PERCENT+TEST_PERCENT:
            with open(os.path.join(ct.TXT_PATH,'test.txt'),'a') as f:
                f.write(img1_path+','+img2_path+','+gt_path)
                f.write('\n')
        else:
            with open(os.path.join(ct.TXT_PATH,'train.txt'),'a') as f:
                f.write(img1_path+','+img2_path+','+gt_path)
                f.write('\n')
        if VISUALIZE:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            gt = Image.open(gt_path)
            gt = np.asarray(gt)*255
            plt.figure(figsize=(200,300))
            plt.subplot(1,3,1)
            plt.imshow(img1)
            plt.subplot(1,3,2)
            plt.imshow(img2)
            plt.subplot(1,3,3)
            plt.imshow(gt)
            plt.show()

class OSCD_TRAIN(Dataset):
    def __init__(self, data_path, dir_nm):
        super(OSCD_TRAIN, self).__init__()
        self.dir_nm = dir_nm
        self.data_path = data_path
        with open(os.path.join(self.dir_nm),'r') as f:
            self.list = f.readlines()
        self.file_size = len(self.list)

    def __getitem__(self, idx):
        x1 = Image.open(os.path.join(self.data_path, self.list[idx].split(',')[0]))
        x2 = Image.open(os.path.join(self.data_path, self.list[idx].split(',')[1]))
        gt = Image.open(os.path.join(self.data_path, self.list[idx].split(',')[2].strip()))
        dir_name = self.list[idx].split(',')[0].split('\\')[0]
#         x1 = x1.filter(ImageFilter.SHARPEN)  
#         x2 = x2.filter(ImageFilter.SHARPEN)  
#         gt = gt.filter(ImageFilter.SHARPEN)  
        
        t = [            
            transforms.RandomRotation((360,360), resample=False, expand=False, center=None),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation((180,180), resample=False, expand=False, center=None),
#             transforms.ColorJitter(brightness=0.8),
#             transforms.ColorJitter(contrast=0.8),
            transforms.Resize((ct.ISIZE,ct.ISIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ]
        if TRANSFORM:
            k = np.random.randint(4)        
            x1 = t[k](x1);x2 = t[k](x2);gt = t[k](gt);
            x1 = t[4](x1);x2 = t[4](x2);gt = t[4](gt);
            x1 = t[5](x1);x2 = t[5](x2);gt = t[5](gt);
            x1 = t[6](x1);x2 = t[6](x2);
#         x1,_ = histeq(x1)
#         x2,_ = histeq(x2)
        if gt.shape[0]==3:
            gt = gt[1, :, :].unsqueeze(0)

        return x1, x2, gt, dir_name

    def __len__(self):
        return self.file_size

class OSCD_TEST(Dataset):
    def __init__(self, data_path, dir_nm):
        super(OSCD_TEST, self).__init__()
        self.dir_nm = dir_nm
        self.data_path = data_path
        with open(os.path.join(self.dir_nm),'r') as f:
            self.list = f.readlines()
        self.file_size = len(self.list)

    def __getitem__(self, idx):
        x1 = Image.open(os.path.join(self.data_path, self.list[idx].split(',')[0]))
        x2 = Image.open(os.path.join(self.data_path, self.list[idx].split(',')[1]))
        gt = Image.open(os.path.join(self.data_path, self.list[idx].split(',')[2].strip()))
        dir_name = self.list[idx].split(',')[0].split('\\')[0]
#         x1 = x1.filter(ImageFilter.SHARPEN)  
#         x2 = x2.filter(ImageFilter.SHARPEN)  
#         gt = gt.filter(ImageFilter.SHARPEN)  
        t = [            
            transforms.Resize((ct.ISIZE,ct.ISIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ]
        if TRANSFORM:       
            x1 = t[0](x1);x2 = t[0](x2);gt = t[0](gt);
            x1 = t[1](x1);x2 = t[1](x2);gt = t[1](gt);
            x1 = t[2](x1);x2 = t[2](x2);
#         x1,_ = histeq(x1)
#         x2,_ = histeq(x2)
        if gt.shape[0]==3:
            gt = gt[1, :, :].unsqueeze(0)
        return x1, x2, gt, dir_name

    def __len__(self):
        return self.file_size

if __name__=='__main__':
    gen_data()
    print('finished')

