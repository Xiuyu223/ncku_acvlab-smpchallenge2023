import os
import gc
import cv2
import math
import copy
import time
import random
from torchvision import models, transforms
# For data manipulation
import numpy as np
import pandas as pd
from PIL import Image
# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
# Albumentations for augmentations
from sklearn.metrics import f1_score, roc_auc_score

import timm
from timm.models.efficientnet import *

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import glob

# os.environ["CUDA_VISIBLE_DEVICES"] = "6,0,3,4,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Number of classes in the dataset
num_classes = 2
# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for
num_epochs = 30
feature_extract = True



def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()

train_df = pd.read_csv('train/done_geo_category_none_old_train_processed_na_time.csv')
# print(train_df.shape, valid_df.shape)

class Dataset(Dataset):

    def __init__(self, df, transform):
        ##############################################
        ### Initialize paths, transforms, and so on
        ### data list -> DataFrame ID, Label
        ##############################################
        self.transform = transform
        #
        # load image path and annotations
        self.df = df
        # self.file_names = df['filename'].values
        self.path = df['img_filepath_local'].values
        # self.labels = df['label'].values
        self.transforms = transforms

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        imgpath = self.path[index]
        img = Image.open(imgpath).convert('RGB')
        # lbl = float(self.labels[index])

        if self.transform is not None:
            img = self.transform(img)
        return {
            'image': img,
            'id': imgpath
        }

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################

        return len(self.df['img_filepath_local'])


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


model = timm.create_model(
    'eva02_large_patch14_448.mim_m38m_ft_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)

model = nn.DataParallel(model, device_ids=[0,1,2,3,4])

model = model.to(device)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ]),
    'val': transforms.Compose([
        transforms.Resize([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
}


@torch.inference_mode()
def get_embeddings(model, dataloader, device):
    model.eval()

    LABELS = []
    EMBEDS = []
    IDS = []
    df_100 = pd.DataFrame()
    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:

            data_img = data['image']
            ids=data['id']
            images = data_img.to(device, dtype=torch.float)
            outputs = model(images)
            #print(outputs.shape)
            for emb_, id_ in zip(outputs.cpu().numpy(), ids):
                data = {'embed': [emb_.flatten().tolist()],
                        'img_path': [id_]}
                df = pd.DataFrame(data)
                df_100 = pd.concat([df_100, df], ignore_index=True)

    return EMBEDS, IDS, df_100


trainset = Dataset(df=train_df, transform=data_transforms['val'])

train_loader = DataLoader(dataset=trainset,
                          batch_size=batch_size,shuffle=False,
                          num_workers=5,pin_memory=True)

_, _, df_100 = get_embeddings(model, train_loader, device)
df_100.to_csv('train/eva_embeddinig_train_25.csv')
import pickle

df_100.to_pickle('train/newtrain_eva_img_embedding.pkl')
print('done')