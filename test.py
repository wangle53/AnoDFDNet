import os
import time
import torch
import numpy as np
import torchvision.utils as vutils
from model import AnoDFDNet
import constants as ct
from make_dataset import OSCD_TEST
from torch.utils.data import Dataset, DataLoader
import evaluate as eva
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_network():
    threshold = ct.THRESHOLD
    test_dir = ct.TEST_TXT
    path = './outputs/bestModel/net.pth'
    epoch = torch.load(path)['epoch']
    pretrained_dict = torch.load(path)['model_state_dict']
    device = torch.device('cpu')
    test_data = OSCD_TEST(ct.DATA_PATH, test_dir)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    net = AnoDFDNet(ct.ISIZE, ct.NC, ct.NZ, ct.NDF, ct.EXTRALAYERS).to(device)
    net.load_state_dict(pretrained_dict,False)
    with net.eval() and torch.no_grad():
        i = 0; TP = 0; FN = 0; FP = 0; TN = 0
        labels = []
        fakes = []
        for i, data in enumerate(test_dataloader):
            x1, x2, gt, dir_name = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            gt = gt.to(device, dtype=torch.float)
            fake = net(x1, x2)
            am = fake
            am = np.array(am.detach().squeeze().cpu().numpy())
            am[am >= ct.THRESHOLD] = 1
            am[am < ct.THRESHOLD] = 0
            labels.append(gt.cpu().numpy())
            fakes.append(fake.cpu().numpy()) 
            save_path = './outputs/test_output'
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            if True:
                vutils.save_image(x1.data, os.path.join(save_path, dir_name[0]+'_x1.png'), normalize=True)
                vutils.save_image(x2.data, os.path.join(save_path, dir_name[0]+'_x2.png'), normalize=True)
                vutils.save_image(fake.data, os.path.join(save_path, dir_name[0]+'_fake.png'), normalize=True)
                vutils.save_image(gt, os.path.join(save_path, dir_name[0]+'_gt.png'), normalize=True)
                cv2.imwrite(os.path.join(save_path, dir_name[0]+'_am.png'), am*255)
                
            tp, fp, tn, fn = eva.confuse_matrix(fake, gt)    
            TP += tp
            FN += fn
            TN += tn
            FP += fp
            i += 1
            print('testing {}th images'.format(i))
    current_auc = eva.roc(labels, fakes, -1, epoch, saveto='./outputs/testROC')
    metrics = eva.eva_metrics(TP, FP, TN, FN)
    with open(os.path.join('./outputs/', 'test_performance.txt'),'a') as f:
        f.write('Time:{},current_epoch:{},iou:{},f1:{},precision:{},recall:{},oa:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), \
                epoch, metrics['iou'],metrics['f1'],metrics['precision'],metrics['recall'],metrics['oa']))
        f.write('\n') 
    for k, v in metrics.items():
        print(k, v)

if __name__ =='__main__':
    test_network()    