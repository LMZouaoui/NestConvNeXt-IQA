import torch
import torchvision.models as models
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor

import numpy as np
from scipy import stats
from tqdm import tqdm
import os
import math
import csv
import copy
import json

import data_loader

from convnext_modify import RDConvNeXt, RDConvNeXt2, RDConvNeXt3, RDConvNeXt4


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            torch.nn.SiLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

#########################################################################

class dualAtten(nn.Module):
    def __init__(self, dim,ffn_expansion_factor=2, bias=True):
        super(dualAtten, self).__init__()
        hidden_feture = int(dim*ffn_expansion_factor)
        self.conv1 = nn.Conv2d(dim, hidden_feture * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(2*hidden_feture, 2*hidden_feture, kernel_size=7, padding=3, groups=2*hidden_feture)  # depthwise conv

        self.CA = CALayer(hidden_feture)
        self.PA = PALayer(hidden_feture)

        self.fuse_conv1 = nn.Conv2d(2*hidden_feture, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x1 = self.conv1(x)
        #
        x_ca,x_pa = self.dwconv(x1).chunk(2, dim=1)
        x_ca = self.CA(x_ca)
        x_pa = self.PA(x_pa)
        x_fuse = torch.cat([x_ca, x_pa], 1)
        out = self.fuse_conv1(x_fuse) # + x

        return out

class DASB(nn.Module):
    def __init__(self, embed_dim,distillation_rate=0.50):
        super(DASB, self).__init__()

        self.distilled_channels = int(embed_dim * distillation_rate)
        self.remaining_channels = int(embed_dim - self.distilled_channels)
        self.distillation_rate = distillation_rate

        self.Conv3_D1 = nn.Conv2d(self.distilled_channels, self.distilled_channels, 3, 1, 1)
        #self.Conv3_D2 = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, 1, 1)

        self.dualAtten = dualAtten(dim = self.distilled_channels)				

    def forward(self, x):

        distilled_c1, remaining_c1 = torch.split(x, (self.distilled_channels, self.remaining_channels), dim=1)
        distilled_c1 = self.Conv3_D1(distilled_c1)

        out1 = self.dualAtten(distilled_c1)
        out = torch.cat([out1, remaining_c1], dim=1)
        x_out = x + out

        return x_out


class L2pooling(nn.Module):
	def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
		super(L2pooling, self).__init__()
		self.padding = (filter_size - 2 )//2
		self.stride = stride
		self.channels = channels
		a = np.hanning(filter_size)[1:-1]
		g = torch.Tensor(a[:,None]*a[None,:])
		g = g/torch.sum(g)
		self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
	def forward(self, input):
		input = input**2
		out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
		return (out+1e-12).sqrt()


class Net(nn.Module):
	def __init__(self): ##, device
		super(Net, self).__init__()
			
		#self.device = device

		#self.cfg = cfg
		self.L2pooling_l1 = L2pooling(channels=256)
		self.L2pooling_l2 = L2pooling(channels=512)
		self.L2pooling_l3 = L2pooling(channels=1024)
		self.L2pooling_l4 = L2pooling(channels=2048)
		""" self.L2pooling_l1 = L2pooling(channels=96)
		self.L2pooling_l2 = L2pooling(channels=192)
		self.L2pooling_l3 = L2pooling(channels=384)
		self.L2pooling_l4 = L2pooling(channels=768) """


		from resnet_modify  import resnet50 as resnet_modifyresnet
		dim_modelt = 3840 #+256*3
		#dim_modelt = 2048
		#dim_modelt = 1024
		modelpretrain = models.resnet50(pretrained=True)
    

		torch.save(modelpretrain.state_dict(), 'modelpretrain')
		
		self.model = resnet_modifyresnet()
		self.model.load_state_dict(torch.load('modelpretrain'), strict=True)		

		self.dim_modelt = dim_modelt

		os.remove("modelpretrain")
		
		self.RDConvNeXt_Nest4 = RDConvNeXt(in_channels = 2048, growth_rate = 2048, num_layers = 1, drop_path=0., layer_scale_init_value=1e-6)
		self.RDConvNeXt_Nest3 = RDConvNeXt2(in_channels = 1024, growth_rate2 = 1024, growth_rate1 = 2048, drop_path=0., layer_scale_init_value=1e-6)
		self.RDConvNeXt_Nest2 = RDConvNeXt3(in_channels = 512, growth_rate3 = 512, growth_rate2 = 1024, drop_path=0., layer_scale_init_value=1e-6)
		self.RDConvNeXt_Nest1 = RDConvNeXt4(in_channels = 256, growth_rate4 = 256, growth_rate3 = 512, drop_path=0., layer_scale_init_value=1e-6)

        
		self.DASB = DASB(embed_dim=dim_modelt, distillation_rate=0.50)

		self.fc = nn.Linear(self.model.fc.in_features + dim_modelt, 1)

		self.avg7 = nn.AvgPool2d((7, 7))
		self.avg8 = nn.AvgPool2d((8, 8))
		self.avg4 = nn.AvgPool2d((4, 4))
		self.avg2 = nn.AvgPool2d((2, 2))
		
			   
		
		self.drop2d = nn.Dropout(p=0.1)
		
	
	def forward(self, x):

		x = torch.cat((x, x, x), dim=1)
		out,layer1,layer2,layer3,layer4 = self.model(x) 

		#print('x', x.shape)        
		#sal = self.SaliencyMap(x)
		#print('sal', sal.shape)

		""" print('out', out.shape)
		print('layer1', layer1.shape)
		print('layer2', layer2.shape)
		print('layer3', layer3.shape)
		print('layer4', layer4.shape) """
            
		layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1,dim=1, p=2))))
		layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2,dim=1, p=2))))
		layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3,dim=1, p=2))))
		layer4_t =           self.drop2d(self.L2pooling_l4(F.normalize(layer4,dim=1, p=2)))
		
		##### 4	
		# 	
		layer41_t =                                  self.RDConvNeXt_Nest4(layer4_t)
		layer32_t, layer31_t =                       self.RDConvNeXt_Nest3(layer3_t, layer41_t)		
		layer23_t, layer22_t, layer21_t =            self.RDConvNeXt_Nest2(layer2_t, layer32_t, layer31_t)
		layer14_t, layer13_t, layer12_t, layer11_t = self.RDConvNeXt_Nest1(layer1_t, layer23_t, layer22_t, layer21_t)
          
		""" print('layer14_t', layer14_t.shape)
		print('layer23_t', layer23_t.shape)
		print('layer32_t', layer32_t.shape)
		print('layer41_t', layer41_t.shape)
		print('layer13_t', layer13_t.shape)
		print('layer12_t', layer12_t.shape)        
		print('layer11_t', layer11_t.shape) """
         
		layers = torch.cat((layer14_t, layer23_t,layer32_t,layer41_t),dim=1)

		
		out_t_c = self.DASB(layers)
		out_t_o = torch.flatten(self.avg7(out_t_c),start_dim=1)

		layer4_o = self.avg7(layer4)
		layer4_o = torch.flatten(layer4_o,start_dim=1)
		            
		predictionQA = self.fc(torch.flatten(torch.cat((out_t_o,layer4_o),dim=1),start_dim=1))          
		return predictionQA
     


def predict_one_image(Network, img, svPath, device, pretrained=1):
    

    if pretrained:
        Network.load_state_dict(
            torch.load(svPath)
        )

    Network.eval()

    with torch.no_grad():
        # 
        img = img.to(device)

        pred = Network(img)              # shape: [N_patches, 1] or [N_patches]
        pred = pred.cpu().numpy()

        # 
        score = np.mean(pred)

    return float(score)