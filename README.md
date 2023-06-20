## ncku_acvlab-smpchallenge2023

### Brief Introduction 
This repository is the code of our **team [ncku_acvlab]** used in SMP Challenge 2023(http://smp-challenge.com/).

We provide all the processed features and necessary codes in this repository.

If you like to directly make the prediction of popularity scores, just clone this repository, jump to step 4 and execute R09 after downloading these necessary features by **this link: https://1drv.ms/u/s!AmaAVPK0zhHXpX7mxw31woqiTbwK?e=6aCqO8**


### Instruction
If you aim to reproduce the whole experiment, please run the code with the following instruction:

1. Based on our previous work, we take the processed features used last year as our baseline. Therefore, the code should be run from R01 to R08 at first, and all the processed features in 'data_2022/train' and 'data_2022/test' would be produced.

For step 2 and 3, we also provide the processed features in 'train/'. If you want to reproduce, please follow step 2 and 3. The processed features are also provide here for download:  

2. This year we adopt more feature extraction methods. To get these features, please run the following code step by step:
   
    2.1. run the python file from 1.py to 13.py
   
    2.2. run BLIP_multimodality_feature/python files from 1.py to 12.py (need to modify output path)
   
3. We also crawled some new features by pathalias. The code is available in 'new_feature_crawler_by_pathalias.py'. Execute to crawl the external data. Also, the organized data is in 'train/new_features.pkl'.

4. With the above steps, we have already finished processing data and features. Run R09 to start training and inference! The result file would be output to **'submission.json'**. As our model is ensembled by lightgbm and tabnet in ratio of 7:3, we also output the result file of these two model, respectively.
  
#### Reminder
- For the part of features same as last year:
We provided the two kinds of the extracted image features which are stored in *.csv format: image captioning and image 
semantic feature. Image captioning information can be extracted by executing R_04 (under tensorflow 2.0). Image semantic feature is extracted by adopting the open source project - TF_FeatureExtraction (https://github.com/tomrunia/TF_FeatureExtraction) on each image.

- In this year, we do image captioning with bilp. It is available in the open source project. If you want to reproduce this part, please follow this repository (https://github.com/salesforce/LAVIS) and build it from source. We used a pretrained blip captioning model trained with coco. We also used sentence_transformers(https://github.com/UKPLab/sentence-transformers) for getting text embedding. Make sure you have install these packages for feature extraction.

- Note that the image and feature files are too large, we didn't put it into our repository. If you want to reproduce the image captioning or image feature extraction part, please put the images to 'imgs/'('imgs/train' and 'imgs/test'). If you want to reproduce or take a look for all the feature processing steps, please download complete file by this link:  

#### Environments
- PC:  i9-9900K, 32GB Memory, Nvidia 3090 Ti.
- OS: Ubuntu 18.04.6 LTS (Bionic Beaver), cuda 11.0
- Software & Libs: Anaconda with python 3.7, Tensorflow 1.12, Tensorflow 2.0 (captioning), pytorch, sklearn, gensim, pandas, lightgbm, and pytorch-tabnet. **You can setup environment with 'requirements.txt'.**

#### Copyright
- Author: Chih-Chung Hsu
e-mail: cchsu@gs.ncku.edu.tw

- Author: Chia-Ming Lee
e-mail: zuw408421476@gmail.com

- Author: Xiu-Yu Hou
e-mail: xiuyu.hou.tw@gmail.com

- Author: Chih-Han Tsai
e-mail: fateplsf567@gmail.com
