

import torch
import torchvision
import torchvision.transforms.functional as F
import folders


class DataLoader(object):
	"""
	Dataset class for IQA databases
	"""

	def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):

		self.batch_size = batch_size
		self.istrain = istrain
				
		if (dataset == 'LWIR_IQA') | (dataset == 'LWIR_IQA_by_class') | (dataset == 'TIIQAD'):
			if istrain:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.Grayscale(num_output_channels=1),
					torchvision.transforms.Resize((512, 512)),
					torchvision.transforms.RandomHorizontalFlip(),
					torchvision.transforms.RandomVerticalFlip(),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					#torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
					#								 std=(0.229, 0.224, 0.225))])
					torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
			else:
				transforms = torchvision.transforms.Compose([
					torchvision.transforms.Grayscale(num_output_channels=1),				
					torchvision.transforms.Resize((512, 512)),
					torchvision.transforms.RandomCrop(size=patch_size),
					torchvision.transforms.ToTensor(),
					#torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
					#								 std=(0.229, 0.224, 0.225))])
					torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

		if dataset == 'LWIR_IQA':
			self.data = folders.LWIRIQAFolder(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)
		elif dataset == 'LWIR_IQA_by_class':
			self.data = folders.LWIRIQABYCLASSFolder(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)			
		elif dataset == 'TIIQAD':
			self.data = folders.TIIQADFolder(
				root=path, index=img_indx, transform=transforms, patch_num=patch_num)	

	def get_data(self):
		if self.istrain:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=self.batch_size, shuffle=True)
		else:
			dataloader = torch.utils.data.DataLoader(
				self.data, batch_size=1, shuffle=False)
		return dataloader