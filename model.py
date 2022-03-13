""" Network architectures.
"""

import torch
import torch.nn as nn
import constants as ct
from _nsis import out
from transformer import Transformer

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

class CNN(nn.Module):
    def __init__(self, isize, nc, nz, ndf, n_extra_layers):
        super(CNN, self).__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            )
        self.e_extra_layers = nn.Sequential()
        for t in range(n_extra_layers):
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(t, ndf),
                            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, ndf),
                            nn.BatchNorm2d(ndf))
            self.e_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(t, ndf),
                            nn.LeakyReLU(0.2, inplace=True))
        self.e2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.e3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.e4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.e5 = nn.Sequential(
            nn.Conv2d(ndf*8, nz, 3, 1, 1, bias=False),
            )
 
    def forward(self,x):
        e1 = self.e1(x)
        e_el = self.e_extra_layers(e1)
        e2 = self.e2(e_el)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        
        return [e1, e_el, e2, e3, e4, e5]
    
class FeatureDifferenceNet(nn.Module):
    def __init__(self, isize, nc, nz, ndf, n_extra_layers):
        super(FeatureDifferenceNet, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        
        self.d5 = nn.Sequential(
            nn.ConvTranspose2d(nz, ndf*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.ReLU(True),
            )
        
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(ndf*8*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(True),
            )
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(ndf*4*2, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU(True),
            )
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(ndf*2*2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.ReLU(True),
            )
        self.d_extra_layers = nn.Sequential()
        for t in range(n_extra_layers):
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-conv'.format(t, ndf*2),
                            nn.Conv2d(ndf*2, ndf, 3, 1, 1, bias=False))
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, ndf*2),
                            nn.BatchNorm2d(ndf))
            self.d_extra_layers.add_module('extra-layers-{0}-{1}-relu'.format(t, ndf*2),
                            nn.ReLU(inplace=True))
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(ndf*2, ct.GT_C, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            )
        self.e5_tr = Transformer()
        
    def forward(self,x1, x2):
        e1 = torch.abs(x1[0]-x2[0])
        e_el = torch.abs(x1[1]-x2[1])
        e2 = torch.abs(x1[2]-x2[2])
        e3 = torch.abs(x1[3]-x2[3])
        e4 = torch.abs(x1[4]-x2[4])
        e5 = torch.abs(x1[5]-x2[5])
    
        B, _, H, W = e5.shape
        if ct.USE_TR:
            e5 = self.e5_tr(e5)
            e5 = e5.permute(0, 2, 1).view(B, ct.HIDDEN_SIZE, H, W)    
        
        d5 = self.d5(e5)
        d4 = self.d4(torch.cat((d5,e4), 1))
        d3 = self.d3(torch.cat((d4, e3), 1))
        d2 = self.d2(torch.cat((d3, e2), 1))
        d_el = self.d_extra_layers(torch.cat((d2, e_el), 1))
        d1 = self.d1(torch.cat((d_el,e1), 1))
    
        return d1

class AnoDFDNet(nn.Module):
    def __init__(self, isize, nc, nz, ndf, n_extra_layers):
        super(AnoDFDNet, self).__init__()
        self.cnn = CNN(isize, nc, nz, ndf, n_extra_layers)
        self.fdn = FeatureDifferenceNet(isize, nc, nz, ndf, n_extra_layers)
        
    def forward(self, x1, x2):
        out1 = self.cnn(x1)
        out2 = self.cnn(x2)
        out = self.fdn(out1, out2)
        
        return out
    
        
 
 
    
if __name__ == '__main__':
    net = AnoDFDNet(ct.ISIZE, ct.NC, ct.NZ, ct.NDF, ct.EXTRALAYERS)
    y = net(torch.randn(2,3,256,256), torch.randn(2,3,256,256))    
    print(y.shape)
        
        
        
        
        
       