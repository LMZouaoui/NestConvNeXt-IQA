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
	def __init__(self): #, cfg, device
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

from torch.optim.lr_scheduler import CosineAnnealingLR

class ConvNeXtEstimat(object):
	
	def __init__(self, config, device,  svPath, datapath, train_idx, test_idx,Net):
		super(ConvNeXtEstimat, self).__init__()
		
		self.device = device
		self.epochs = config.epochs
		self.test_patch_num = config.test_patch_num
		self.l1_loss = torch.nn.L1Loss()
		self.lr = config.lr
		self.lrratio = 10
		self.weight_decay = config.weight_decay
		self.net = Net().to(device) ## config,device
		self.config = config
		self.clsloss =  nn.CrossEntropyLoss()
		self.paras = [{'params': self.net.parameters(), 'lr': self.lr} ]
		self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        ################### Scheduler with T_max = 50 epochs and eta_min=0
		self.scheduler = CosineAnnealingLR(self.solver, T_max=50, eta_min=0)  #################

		train_loader = data_loader.DataLoader(config.dataset, datapath, 
											  train_idx, config.patch_size, 
											  config.train_patch_num, 
											  batch_size=config.batch_size, istrain=True)
		
		test_loader = data_loader.DataLoader(config.dataset, datapath,
											 test_idx, config.patch_size,
											 config.test_patch_num, istrain=False)
		
		self.train_data = train_loader.get_data()
		self.test_data = test_loader.get_data()


	def train(self,seed,svPath):
		best_srcc = 0.0
		best_plcc = 0.0
		print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tLearning_Rate')
		steps = 0
		results = {}
		performPath = svPath +'/' + 'PLCC_SRCC_'+str(self.config.vesion)+'_'+str(seed)+'.json'
		with open(performPath, 'w') as json_file2:
			json.dump(  {} , json_file2)
		
		for epochnum in range(self.epochs):
			self.net.train()
			epoch_loss = []
			pred_scores = []
			gt_scores = []
			pbar = tqdm(self.train_data, leave=False)

			for img, label in pbar:
				img = torch.as_tensor(img.to(self.device)).requires_grad_(False)
				label = torch.as_tensor(label.to(self.device)).requires_grad_(False)

				steps+=1
				
				self.net.zero_grad()
				pred = self.net(img)

				pred_scores = pred_scores + pred.flatten().cpu().tolist()
				gt_scores = gt_scores + label.cpu().tolist()

				loss_qa = self.l1_loss(pred.squeeze(), label.float().detach())

				loss = 10 * loss_qa 
				
				epoch_loss.append(loss.item())
				loss.backward()
				self.solver.step()
				
            ###### Update learning rate
			self.scheduler.step()
	
			modelPath = svPath + '/model_{}_{}_{}'.format(str(self.config.vesion),str(seed),epochnum)
			torch.save(self.net.state_dict(), modelPath)

			train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

			test_srcc, test_plcc = self.test(self.test_data,epochnum,svPath,seed)


			results[epochnum]=(test_srcc, test_plcc)
			with open(performPath, "r+") as file:
				data = json.load(file)
				data.update(results)
				file.seek(0)
				json.dump(data, file)
			

		
			if test_srcc > best_srcc:
				modelPathbest = svPath + '/bestmodel_{}_{}'.format(str(self.config.vesion),str(seed))
				
				torch.save(self.net.state_dict(), modelPathbest)

				best_srcc = test_srcc
				best_plcc = test_plcc

			print('{}\t{:4.3f}\t\t{:4.4f}\t\t{:4.4f}\t\t{:4.3f}\t\t{}'.format(epochnum + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc,self.paras[0]['lr'] ))


		print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

		return best_srcc, best_plcc

	def test(self, data,epochnum,svPath,seed,pretrained=0):
		if pretrained:
			self.net.load_state_dict(torch.load(svPath+'/bestmodel_{}_{}'.format(str(self.config.vesion),str(seed))))
		self.net.eval()
		pred_scores = []
		gt_scores = []
		
		pbartest = tqdm(data, leave=False)

		with torch.no_grad():
			steps2 = 0
					
			for img, label in pbartest:
				img = torch.as_tensor(img.to(self.device))
				label = torch.as_tensor(label.to(self.device))

				pred = self.net(img)
                            
				pred_scores = pred_scores + pred.cpu().tolist()
				gt_scores = gt_scores + label.cpu().tolist()
				
				steps2 += 1
				
		
		pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
		gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
		
# 		if not pretrained:
		dataPath = svPath + '/test_prediction_gt_{}_{}_{}.csv'.format(str(self.config.vesion),str(seed),epochnum)
		with open(dataPath, 'w') as f:
			writer = csv.writer(f)
			writer.writerows(zip(pred_scores, gt_scores))
	
		test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
		test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
		return test_srcc, test_plcc
	
if __name__=='__main__':
	import os
	import argparse
	import random
	import numpy as np
	from args import *
	
	