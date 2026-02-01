
import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
#from openpyxl import load_workbook
import pandas as pd

class LWIRIQAFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []

        csv_file = os.path.join(root, 'scores.csv')

        # Read the CSV file
        df = pd.read_csv(csv_file, delimiter=';')  # use '\t' for tab-separated values
        # Extract the columns of interest
        imgname = df['Distorted_image']
        mos = df['MOS']
        mos_all = np.array(mos).astype(np.float32)
        
        #print('*im*', imgname.size)
        #print('*mos_all*', mos_all)
        
        """ with open(csv_file) as f:
            reader = csv.DictReader(f)

            for row in reader:
                
                imgname.append(row['Distorted_image'].split('/')[1])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)
        
         """
        sample = []

        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length

class LWIRIQABYCLASSFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []

        csv_file = os.path.join(root, 'scores_nu.csv')

        # Read the CSV file
        df = pd.read_csv(csv_file, delimiter=';')  # use '\t' for tab-separated values
        # Extract the columns of interest
        imgname = df['Distorted_image']
        mos = df['MOS']
        mos_all = np.array(mos).astype(np.float32)
        
        #print('*im*', imgname.size)
        #print('*mos_all*', mos_all)
        
        """ with open(csv_file) as f:
            reader = csv.DictReader(f)

            for row in reader:
                
                imgname.append(row['Distorted_image'].split('/')[1])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)
        
         """
        sample = []

        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length
    

class TIIQADFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []

        csv_file = os.path.join(root, 'scores.csv')

        # Read the CSV file
        df = pd.read_csv(csv_file, delimiter=';')  # use '\t' for tab-separated values
        # Extract the columns of interest
        imgname = df['Distorted_image']
        mos = df['MOS']
        mos_all = np.array(mos).astype(np.float32)
        
        #print('*im*', imgname.size)
        #print('*mos_all*', mos_all)
        
        """ with open(csv_file) as f:
            reader = csv.DictReader(f)

            for row in reader:
                
                imgname.append(row['Distorted_image'].split('/')[1])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                mos_all.append(mos)
        
         """
        sample = []

        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root,'images/', imgname[item]), mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')