import os
import time
import yaml
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import cv2 as cv
from PIL import Image

class CelebFacesDataset(Dataset):
    def __init__(self, images_dir, images_names, labels, transforms=None):
        self.images_dir = images_dir
        self.images_names = images_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx):
        image_name = self.images_names[idx]
        image = Image.open(os.path.join(self.images_dir, image_name))
        if self.transforms is not None:
            image = self.transforms(image)

        label = np.array(self.labels.loc[image_name])
        
        return image, label

    def __len__(self):
        return len(self.images_names)


def get_dataloader(batch_size, images_dir, images_names, labels, num_workers=0, train_mode=True, transforms=None):
    if transforms is None:
        transforms = torchvision.transforms.ToTensor()

    dataset = CelebFacesDataset(images_dir=images_dir, images_names=images_names, labels=labels, transforms=transforms)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=train_mode,
                            shuffle=train_mode)
    return dataloader