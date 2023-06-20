'''
Before running this file, please modify the output path to an available path.
'''

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

from models.blip import blip_feature_extractor
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import BertTokenizer

image_size = 448
train_df = pd.read_csv("../train/done_geo_category_none_old_train_processed_na_time.csv",encoding='utf-8')
train_df['Title']=train_df['Title'].fillna('NONE')
train_df = train_df.dropna(subset=['img_filepath_local'])
batch_size=1
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
        self.texts = df['Title'].tolist()
        self.paths = df['img_filepath_local'].tolist()
        # self.labels = df['label'].values
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.transforms = transforms

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        imgpath = self.paths[index]
        img = Image.open(imgpath).convert('RGB')
        # lbl = float(self.labels[index])
        text = self.texts[index]
        text_input = self.process_text(text)
        #print(text)
        if self.transform is not None:
            img = self.transform(img)
        return {
            'image': img,
            'id': imgpath,
            'text':text
        }

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################

        return len(self.df['img_filepath_local'])


    def process_text(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask



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

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth'
model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='large')
model.eval()
model = nn.DataParallel(model, device_ids=[0])
model = model.to(device)


trainset = Dataset(df=train_df, transform=data_transforms['val'])

train_loader = DataLoader(dataset=trainset,
                          batch_size=batch_size,shuffle=False,
                          num_workers=5,pin_memory=True)
                          
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
            data_text=data['text']

            ids=data['id']

            images = data_img.to(device, dtype=torch.float)
            #texts= data_text.to(device)


            outputs = model(images, data_text, mode='text')[0,0]

            #print(outputs.shape)
            for emb_, id_ in zip(outputs.cpu().numpy(), ids):
                data = {'embed': [emb_.flatten().tolist()],
                        'img_path': [id_]}
                df = pd.DataFrame(data)
                df_100 = pd.concat([df_100, df], ignore_index=True)

    return EMBEDS, IDS, df_100

'''
multimodal_feature = model(image, caption, mode='multimodal')[0,0]
image_feature = model(image, caption, mode='image')[0,0]
text_feature = model(image, caption, mode='text')[0,0]
'''
_, _, df_100 = get_embeddings(model, train_loader, device)
df_100.to_csv('/ssd8/ming/flickr/new_old_train_combine/geo_alias/BLIP_embeddinig_train_title_img_text.csv')
import pickle

df_100.to_pickle('/ssd8/ming/multi_lang_emb/new_pickle/train/LARGE_newtrain_BLIP_embeddinig_train_title_img_text.pkl')
print('done')