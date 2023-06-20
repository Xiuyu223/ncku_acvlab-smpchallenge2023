'''
To Run this file, please install the package from source (https://github.com/salesforce/LAVIS) so that the blip model would be available.
'''

from PIL import Image
import requests
import torch
from lavis.models import load_model_and_preprocess
import os
import pandas as pd
import numpy as np

# load captioning model - blip
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
print('Successfully loaded model!')

# train imgs
df = pd.read_csv('train/train_img_all_list.txt', delimiter=' ')
df.loc[:, 'Caption'] = 'Nothing at all'

for i, row in enumerate(df.values):
    img_path = row[2]
    image = Image.open(img_path)
    image = image.resize((384,384))
    image = image.convert('RGB')
    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    cap = model.generate({"image": image})
    
    df.loc[i, 'Caption'] = cap[0]
    if i % 1000 == 0:
        print('row:', i, '\tcaption:', cap[0], '\timg_path:', img_path)
print('Finish captioning for training imgs!')

df = df.drop('Path', axis=1)
df.to_csv('train/train_img_cap_list_blip.txt', index = False)
print('Successfully save captions for training imgs!')

# test imgs
df = pd.read_csv('test/test_img_all_list.txt', delimiter=' ')
df.loc[:, 'Caption'] = 'Nothing at all'

for i, row in enumerate(df.values):
    img_path = row[2]
    image = Image.open(img_path)
    image = image.resize((384,384))
    image = image.convert('RGB')
    image = vis_processors["eval"](image).unsqueeze(0).to(device)

    cap = model.generate({"image": image})
    
    df.loc[i, 'Caption'] = cap[0]
    if i % 1000 == 0:
        print('row:', i, '\tcaption:', cap[0], '\timg_path:', img_path)
print('Finish captioning for testing imgs!')

df = df.drop('Path', axis=1)
df.to_csv('test/test_img_cap_list_blip.txt', index = False)
print('Successfully save captions for testing imgs!')