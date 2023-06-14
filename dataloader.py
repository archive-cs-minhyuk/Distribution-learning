import os
import ast
import torch
import glob
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torchvision import transforms
import torchvision.transforms.functional as F 
from torch.utils.data import Dataset
from PIL import Image

class SingleDataset(Dataset):
    def __init__(self, adm2name, adm3name, metadata, root_album, transform=None):
        self.metadata = pd.read_csv(metadata, index_col=0)
        self.adm2name = adm2name
        self.adm3name = adm3name
        self.root_album = root_album
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        temp = self.metadata[(self.metadata['adm2']==self.adm2name) & (self.metadata['adm3']==self.adm3name)]
        myprovince = temp.values[0][0]
        mycity = self.adm2name
        myadm3 = self.adm3name
        
        if myprovince == 'None':
            image_root_path = "{}/{}/{}".format(self.root_album, mycity, myadm3)
        else:
            image_root_path = "{}/{}/{}".format(self.root_album, myprovince, myadm3)
        images = np.stack([io.imread("{}/{}".format(image_root_path, x)) / 255.0 for x in os.listdir(image_root_path)])    
        sample = {'images': images, 'prov': myprovince, 'city': mycity, 'adm3': myadm3}
        
        if self.transform:
            sample['images'] = self.transform(sample['images']).squeeze()

        return sample

class OproxyDataset(Dataset):
    def __init__(self, metadata, root_dir, transform = None):
        self.metadata = pd.read_csv(metadata,index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_id, y_urban, y_rural, y_env = self.metadata.iloc[idx, :].values
        image_path = "{}{}.png".format(self.root_dir, int(image_id))
        image = io.imread(image_path) / 255.0
        
        if self.transform:
            image = self.transform(np.stack([image])).squeeze()
            
        return image, torch.Tensor([y_urban, y_rural, y_env])   
    
class RandomRotate(object):
    def __call__(self, images):
        rotated = np.stack([self.random_rotate(x) for x in images])
        return rotated
    
    def random_rotate(self, image):
        rand_num = np.random.randint(0, 4)
        if rand_num == 0:
            return np.rot90(image, k=1, axes=(0, 1))
        elif rand_num == 1:
            return np.rot90(image, k=2, axes=(0, 1))
        elif rand_num == 2:
            return np.rot90(image, k=3, axes=(0, 1))   
        else:
            return image

class Grayscale(object):
    def __init__(self, prob = 1):
        self.prob = prob

    def __call__(self, images):     
        random_num = np.random.randint(100, size=1)[0]
        if random_num <= self.prob * 100:
            gray_images = (images[:, 0, :, :] + images[:, 1, :, :] + images[:, 2, :, :]) / 3
            gray_scaled = gray_images.unsqueeze(1).repeat(1, 3, 1, 1)
            return gray_scaled
        else:
            return images        
        
class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, images):
        normalized = np.stack([F.normalize(x, self.mean, self.std, self.inplace) for x in images]) 
        return normalized
        
class ToTensor(object):
    def __call__(self, images):
        images = images.transpose((0, 3, 1, 2))
        return torch.from_numpy(images).float() 