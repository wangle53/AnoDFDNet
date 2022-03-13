#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import os
import torch
import torch.nn as nn
import time
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from make_dataset import gen_data
from make_dataset import gen_data,OSCD_TRAIN,OSCD_TEST
import constants as ct
import evaluate as eva
from loss import l1_loss
from model import AnoDFDNet, weights_init
import utils 
import visdom
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
vis = visdom.Visdom(server="http://localhost", port=8097)

def train_network():
    best_auc = 0
    init_epoch = 0
    best_metric = 0
    total_steps = 0
    train_dir = ct.TRAIN_TXT
    val_dir = ct.VAL_TXT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    train_data = OSCD_TRAIN(ct.DATA_PATH, train_dir)
    train_dataloader = DataLoader(train_data, batch_size=ct.BATCH_SIZE, shuffle=True)
    val_data = OSCD_TEST(ct.DATA_PATH, val_dir)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
    net = AnoDFDNet(ct.ISIZE, ct.NC, ct.NZ, ct.NDF, ct.EXTRALAYERS).to(device=device)
    net.apply(weights_init)
    
    if ct.RESUME:
        assert os.path.exists(os.path.join(ct.WEIGHTS_SAVE_DIR, 'current_net.pth')) \
                and os.path.exists(os.path.join(ct.WEIGHTS_SAVE_DIR, 'current_net.pth')), \
                'There is not found any saved weights'
        print("\nLoading pre-trained networks.")
        init_epoch = torch.load(os.path.join(ct.WEIGHTS_SAVE_DIR, 'current_net.pth'))['epoch']
        net.load_state_dict(torch.load(os.path.join(ct.WEIGHTS_SAVE_DIR, 'current_net.pth'))['model_state_dict'])
        with open('./outputs/val_metrics.txt') as f:
            lines = f.readlines()
            best_metric = float(lines[-1].strip().split(':')[-1])
        print("\tDone.\n")
        
    l_bce = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=ct.LR, betas=(0.5, 0.999))

    start_time = time.time()
    for epoch in range(init_epoch+1, ct.EPOCH):
        loss = []
        net.train()
        epoch_iter = 0
        for i, data in enumerate(train_dataloader):
            x1, x2, gt, dir_name = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            gt = gt.to(device, dtype=torch.float)
            epoch_iter += ct.BATCH_SIZE
            total_steps += ct.BATCH_SIZE
             
            #forward
            fake = net(x1, x2)           
            err = l_bce(fake, gt)
            
            #backward
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
             
            errors = utils.get_errors(err)            
            loss.append(err.item())
            counter_ratio = float(epoch_iter) / len(train_dataloader.dataset)
            if(i%ct.PRINT_STEP==0 and i>0):
                print('epoch:',epoch,'iteration:',i,' loss is {}'.format(np.mean(loss[-(ct.PRINT_STEP+1):])))
                if ct.DISPLAY:
                    utils.plot_current_errors(epoch, counter_ratio, errors, vis)
                    utils.display_current_images(gt.data, fake.data, vis)
        utils.save_current_images(epoch, gt.data, fake.data, ct.IM_SAVE_DIR)
         
        with open(os.path.join('./outputs/','train_loss.txt'),'a') as f:
            f.write('after %s epoch, loss is %g'%(epoch,np.mean(loss)))
            f.write('\n')
        if not os.path.exists(ct.WEIGHTS_SAVE_DIR):
            os.makedirs(ct.WEIGHTS_SAVE_DIR)
        utils.save_weights(epoch, net, optimizer, ct.WEIGHTS_SAVE_DIR, 'net')
        duration = time.time()-start_time
        print('training duration is %g'%duration)

        #val phase
        print('Validating.................')
        with net.eval() and torch.no_grad(): 
            TP = 0; FN = 0; FP = 0; TN = 0;
            labels = []
            fakes = []
            for k, data in enumerate(val_dataloader):
                x1, x2, label, _ = data
                x1 = x1.to(device, dtype=torch.float)
                x2 = x2.to(device, dtype=torch.float)
                label = label.to(device, dtype=torch.float)

                time_i = time.time()
                v_fake= net(x1, x2)
                labels.append(label.cpu().numpy())
                fakes.append(v_fake.cpu().numpy()) 
                tp, fp, tn, fn = eva.confuse_matrix(v_fake, label)    
                TP += tp
                FN += fn
                TN += tn
                FP += fp
            current_auc = eva.roc(labels, fakes, best_auc, epoch)
            if current_auc > best_auc:
                best_auc = current_auc
            metrics = eva.eva_metrics(TP, FP, TN, FN)
            cur_metric = metrics[ct.CRITERIA]
            if not os.path.exists('./outputs/bestModel'):
                os.makedirs('./outputs/bestModel')
            if cur_metric > best_metric: 
                best_metric = cur_metric
                shutil.copy('./outputs/model/current_net.pth','./outputs/bestModel/net.pth')           
            with open(os.path.join('./outputs/','val_metrics.txt'),'a') as f:
                f.write('Time:{},current_epoch:{},criteria:{}, current_metric:{},best_metrci:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), \
                                                                                                          epoch, ct.CRITERIA, cur_metric, best_metric))
                f.write('\n')   
            with open(os.path.join('./outputs/', 'val_performance.txt'),'a') as f:
                f.write('Time:{},current_epoch:{},iou:{},f1:{},precision:{},recall:{},oa:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), \
                        epoch, metrics['iou'],metrics['f1'],metrics['precision'],metrics['recall'],metrics['oa']))
                f.write('\n')  
            print('{}:  current metric {}, best metric {}'.format(ct.CRITERIA, cur_metric, best_metric))
            
if __name__ == '__main__':
#     gen_data()
    train_network()