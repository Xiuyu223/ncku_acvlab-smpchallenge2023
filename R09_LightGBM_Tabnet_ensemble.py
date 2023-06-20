'''
Directly execute this file 'python R09_LightGBM_Tabnet_ensemble.py', then the training and inference would be done.
The submission file will be saved as 'submission.json' after training and inference.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_absolute_error as MAE
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import numpy as np
from gensim.models import word2vec
from gensim.corpora.dictionary import Dictionary
import pickle
import time, datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

import torch
from pytorch_tabnet.tab_model import TabNetRegressor

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
# import gc
from sklearn.model_selection import GridSearchCV
# import jieba, pdb
from sklearn.decomposition import PCA
from scipy import stats
import json
from DataProcessing_test import *

# import matplotlib.pylab as plt
# from gplearn import genetic
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

'''
LightGBM
'''

df_tags = pd.read_pickle('data_2022/train/tags_processed_all.pkl')
df_cat  = pd.read_pickle('data_2022/train/cat_processed_all.pkl')
df_add  = pd.read_pickle('data_2022/train/add_processed_all.pkl')
df_ts   = pd.read_pickle('data_2022/train/ts_processed_all.pkl')
df_cap  = pd.read_pickle('data_2022/train/cap_processed_all.pkl')
df_label= pd.read_pickle('data_2022/train/label_processed_all.pkl')

df_userdata= pd.read_pickle('train/user_data_processed_all.pkl')
df_new_feats = pd.read_pickle('train/newfeatures_processed_all.pkl')


tfidf_cap = np.load('data_2022/train/tfidf_cap.npy')
dd = np.load('data_2022/train/tfidf_cat.npz')
cat1, cat2, cat3 = dd['cat1'],dd['cat2'],dd['cat3']
del dd


print("Got %d training with %d labels" % (len(df_cat), len(df_label)))

df_tags = df_tags.drop(['Unnamed: 0'], axis=1)
# df_imfeat = df_imfeat.drop(['Unnamed: 0'], axis=1)

df_all = df_cat.merge(df_tags, on=['Uid', 'Pid'])
df_all = df_ts.merge(df_all, on=['Uid', 'Pid'])
df_all = df_add.merge(df_all, on=['Uid', 'Pid'])
df_all = df_cap.merge(df_all, on=['Uid', 'Pid'])

df_all = df_userdata.merge(df_all, on=['Uid', 'Pid'])
# df_all = pd.concat([df_all, df_capemo], axis=1)
df_all = df_new_feats.merge(df_all, on=['Uid', 'Pid'])


ss_len = len(df_all.columns)

# df_all = df_all.merge(df_imfeat, on=['Uid', 'Pid'], how='outer')
# df_all=  df_all.drop(['Uid', 'Pid'], axis=1)


tfidf_cap = np.load('data_2022/train/tfidf_cap_pca.npy')


catf = []

for i, c in enumerate(df_all.columns):
    print(f"[{i}]: {c}: {df_all[c].dtype}")
    if str(df_all[c].dtype)=='object':
        print(f"[{i}]: {c}: {df_all[c].dtype}")
        catf.append(c)

df_all = df_all.drop(catf, axis=1)
df_train = df_all.iloc[0:305613]
df_test  = df_all.iloc[305613:]

# train/test data
print("Got %d training with %d labels" % (len(df_all), len(df_label)))


imf =np.load('data_2022/train/imfpca.npy')

dd = np.load('data_2022/train/roberta.npz')
roberta1 = dd['roberta1']
roberta2 = dd['roberta2']

dd = np.load('data_2022/train/tfidf_cat_pca.npz')
cat1, cat2, cat3 = dd['cat1'],dd['cat2'],dd['cat3']
del dd


df_folds = pd.read_csv('uid_fold.csv')
df_train_fold = pd.merge(df_train['Uid'], df_folds, on='Uid', how='left')


user_description = np.load('train/user_description_pca50.npy')


multi_lang_emb_tag = np.load('train/Alltags_embedding_all_pca20.npy')
multi_lang_emb_title = np.load('train/Title_embedding_all_pca20.npy')
cap_emb = np.load('train/Caption_embedding_blip_all_pca50.npy')
img_emb_eva = np.load('train/eva_img_embedding_all_pca100.npy')





X = df_train.values

X=X[:,:ss_len]
X= np.concatenate((X, imf[0:305613,:]),axis=1)
X= np.concatenate((X, roberta2[0:305613,:]),axis=1)
X= np.concatenate((X, roberta1[0:305613,:]),axis=1)
X = np.concatenate((X, tfidf_cap[:305613,:]),axis=1)
X = np.concatenate((X, cat1[:305613,:]),axis=1)
X = np.concatenate((X, cat2[:305613,:]),axis=1)
X = np.concatenate((X, cat3[:305613,:]),axis=1)
X = np.concatenate((X, user_description[:305613,:]),axis=1)
X = np.concatenate((X, multi_lang_emb_tag[:305613,:]),axis=1)
X = np.concatenate((X, multi_lang_emb_title[:305613,:]),axis=1)
X = np.concatenate((X, cap_emb[:305613,:]),axis=1)
X = np.concatenate((X, img_emb_eva[:305613,:]),axis=1)

print('X shape:', X.shape)
Y = df_label.values
Y = np.reshape(Y,[-1])
Y2=np.asarray(Y,np.int)


X2 = df_test.values

X2=X2[:,:ss_len]
X2= np.concatenate((X2, imf[305613:,:]),axis=1)
X2= np.concatenate((X2, roberta2[305613:,:]),axis=1)
X2= np.concatenate((X2, roberta1[305613:,:]),axis=1)
X2 = np.concatenate((X2, tfidf_cap[305613:,:]),axis=1)
X2 = np.concatenate((X2, cat1[305613:,:]),axis=1)
X2 = np.concatenate((X2, cat2[305613:,:]),axis=1)
X2 = np.concatenate((X2, cat3[305613:,:]),axis=1)
X2 = np.concatenate((X2, user_description[305613:,:]),axis=1)
X2 = np.concatenate((X2, multi_lang_emb_tag[305613:,:]),axis=1)
X2 = np.concatenate((X2, multi_lang_emb_title[305613:,:]),axis=1)
X2 = np.concatenate((X2, cap_emb[305613:,:]),axis=1)
X2 = np.concatenate((X2, img_emb_eva[305613:,:]),axis=1)

print('Test data shape:', X2.shape)
# del roberta1, roberta2, imf

oof_preds = np.zeros(X.shape[0])
feature_importance_df = pd.DataFrame()
t_value = np.zeros(X2.shape[0])
t_cnt = 0

# train_x, train_y = X[train_list,:], Y[train_list]


# # train_x, train_y = X, Y
# valid_x, valid_y = X[val_list,:], Y[val_list]




params = {
'nthread': 16, 'boosting_type': 'dart', 'objective': 'regression', 'metric': ['mae','mse'], 
    'learning_rate': 0.02, 'num_leaves': 128,
'max_depth': 7, 'subsample': 0.7, 'feature_fraction': 0.9, 
    'min_split_gain': 0.09, 'min_child_weight': 9.5,
'drop_rate':0.8, 'skip_drop':0.8, 'max_drop': 50, 'uniform_drop':False, 
    'xgboost_dart_mode':True, 'drop_seed':1  }


bsts = []
t_value = np.zeros((X2.shape[0]))

t_values = np.zeros((X2.shape[0], 5))
t_value_median = np.zeros((X2.shape[0]))

folds = [1,2,3,4,5]

list_Ytest =[]
list_pred =[]

mae_scores = []
mse_scores = []
spearman_scores = []

start_time = time.time()

for fold in folds:# train_index, test_index , kf.split(X)
    train_index = df_train_fold.index[df_train_fold['Fold']!=fold]
    test_index = df_train_fold.index[df_train_fold['Fold']==fold]
    print("TRAIN:", train_index, "TEST:", test_index)
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    dtrain = lgb.Dataset(X[train_index], label=Y[train_index])
    dval = lgb.Dataset(X[test_index], label=Y[test_index], reference=dtrain)
    bst = lgb.train(params, dtrain, num_boost_round=21500, valid_sets=[dval], early_stopping_rounds=200, verbose_eval=100)
    bsts.append(bst)

    predictions = bst.predict(X[test_index], num_iteration=21500)
    
    list_Ytest.append(Y[test_index])
    list_pred.append(predictions)
    
    
    t_value_median = np.expand_dims(bst.predict(X2, num_iteration=21500), axis=1)
    t_values[:, fold-1] = t_value_median[:, 0]



for validY, pred in zip(list_Ytest, list_pred):
    print('SRC: %.6f' % (stats.spearmanr(validY, pred)[0]), end=', ')#%d-iter, bst.best_iteration, 
    print('MAE : %.6f, MSE : %.6f' % (mean_absolute_error(validY, pred), mean_squared_error(validY, pred)))


allYtest = np.concatenate(list_Ytest)
allpreds = np.concatenate(list_pred)
print('[CV]')
print('SRC: %.6f' % (stats.spearmanr(allYtest, allpreds)[0]), end=', ')
print('MAE : %.6f, MSE : %.6f' % (mean_absolute_error(allYtest, allpreds), mean_squared_error(allYtest, allpreds)))

    
t_value = np.mean(t_values, axis=1)
t_value_median = np.median(t_values, axis=1)


end_time = time.time()
total_time = end_time - start_time
print("5-fold executed time:", total_time, "s")

#save mean result
order = t_value.argsort()
ranks = order.argsort()
ranks = max(ranks) - ranks+1
df_cat = pd.read_json("data_2022/test/test_category.json")

y=pd.DataFrame(t_value)
y.columns=['y']
df_res = df_cat.join(y)
df_res=df_res[['Pid','y']]
df_res=df_res.astype({'Pid':int})
df_res.columns=['post_id','popularity_score']
res_key = pd.DataFrame(['result' for i in range(len(df_res))])
df_res = df_res.join(res_key)

df_res.columns=['post_id','popularity_score', 'result']
df_res['post_id'] = df_res['post_id'].astype('str')

j = '{"version": "VERSION 1.2","result":['

for ind, (x,y) in enumerate(df_res[['post_id','popularity_score']].values):
    if ind == len(df_res)-1:
        j += '{"post_id": "%s","popularity_score": %f}' % (x,y)
    else:
        j += '{"post_id": "%s","popularity_score": %f},' % (x,y)
    
j+=']}'
with open('submission-lgbm.json','w') as fp:
    fp.write(j)

print('Good job!')


'''
Tabnet
'''
df_tags = pd.read_pickle('data_2022/train/tags_processed_all.pkl')
df_cat  = pd.read_pickle('data_2022/train/cat_processed_all.pkl')
df_add  = pd.read_pickle('data_2022/train/add_processed_all.pkl')
df_ts   = pd.read_pickle('data_2022/train/ts_processed_all.pkl')
df_cap  = pd.read_pickle('data_2022/train/cap_processed_all.pkl')
df_label= pd.read_pickle('data_2022/train/label_processed_all.pkl')

df_userdata= pd.read_pickle('train/user_data_processed_all.pkl') 
df_new_feats = pd.read_pickle('train/newfeatures_processed_all.pkl')


tfidf_cap = np.load('data_2022/train/tfidf_cap.npy')
dd = np.load('data_2022/train/tfidf_cat.npz')
cat1, cat2, cat3 = dd['cat1'],dd['cat2'],dd['cat3']
del dd


print("Got %d training with %d labels" % (len(df_cat), len(df_label)))

df_tags = df_tags.drop(['Unnamed: 0'], axis=1)

df_all = df_cat.merge(df_tags, on=['Uid', 'Pid'])
df_all = df_ts.merge(df_all, on=['Uid', 'Pid'])
df_all = df_add.merge(df_all, on=['Uid', 'Pid'])
df_all = df_cap.merge(df_all, on=['Uid', 'Pid'])

df_all = df_userdata.merge(df_all, on=['Uid', 'Pid'])
df_all = df_new_feats.merge(df_all, on=['Uid', 'Pid'])


ss_len = len(df_all.columns)


tfidf_cap = np.load('data_2022/train/tfidf_cap_pca.npy')


catf = []

for i, c in enumerate(df_all.columns):
    print(f"[{i}]: {c}: {df_all[c].dtype}")
    if str(df_all[c].dtype)=='object':
        print(f"[{i}]: {c}: {df_all[c].dtype}")
        catf.append(c)

df_all = df_all.drop(catf, axis=1)
df_train = df_all.iloc[0:305613]
df_test  = df_all.iloc[305613:]

# train/test data
print("Got %d training with %d labels" % (len(df_all), len(df_label)))


imf =np.load('data_2022/train/imfpca.npy')

dd = np.load('data_2022/train/roberta.npz')
roberta1 = dd['roberta1']
roberta2 = dd['roberta2']

dd = np.load('data_2022/train/tfidf_cat_pca.npz')
cat1, cat2, cat3 = dd['cat1'],dd['cat2'],dd['cat3']
del dd


df_folds = pd.read_csv('uid_fold.csv')
df_train_fold = pd.merge(df_train['Uid'], df_folds, on='Uid', how='left')


user_description = np.load('train/user_description_pca50.npy')  


multi_lang_emb_tag = np.load('train/Alltags_embedding_all_pca20.npy')
multi_lang_emb_title = np.load('train/Title_embedding_all_pca20.npy')
cap_emb = np.load('train/Caption_embedding_blip_all_pca50.npy')
img_emb_eva = np.load('train/eva_img_embedding_all_pca100.npy')



blip_tags_img_img = np.load('train/BLIP_embeddinig_all_alltag_img_img.npy').reshape((-1, 1))
blip_tags_img_mutimodal = np.load('train/BLIP_embeddinig_all_alltag_img_multimodal.npy').reshape((-1, 1))
blip_tags_img_text = np.load('train/BLIP_embeddinig_all_alltag_img_text.npy').reshape((-1, 1))
blip_title_img_img = np.load('train/BLIP_embeddinig_all_title_img_img.npy').reshape((-1, 1))
blip_title_img_mutimodal = np.load('train/BLIP_embeddinig_all_title_img_multimodal.npy').reshape((-1, 1))
blip_title_img_text = np.load('train/BLIP_embeddinig_all_title_img_text.npy').reshape((-1, 1))


X = df_train.values

X=X[:,:ss_len]
X= np.concatenate((X, imf[0:305613,:]),axis=1)
X= np.concatenate((X, roberta2[0:305613,:]),axis=1)
X= np.concatenate((X, roberta1[0:305613,:]),axis=1)
X = np.concatenate((X, tfidf_cap[:305613,:]),axis=1)
X = np.concatenate((X, cat1[:305613,:]),axis=1)
X = np.concatenate((X, cat2[:305613,:]),axis=1)
X = np.concatenate((X, cat3[:305613,:]),axis=1)
X = np.concatenate((X, user_description[:305613,:]),axis=1)
X = np.concatenate((X, multi_lang_emb_tag[:305613,:]),axis=1)
X = np.concatenate((X, multi_lang_emb_title[:305613,:]),axis=1)
X = np.concatenate((X, cap_emb[:305613,:]),axis=1)
X = np.concatenate((X, img_emb_eva[:305613,:]),axis=1)

X = np.concatenate((X, blip_tags_img_img[:305613,:]),axis=1)
X = np.concatenate((X, blip_tags_img_mutimodal[:305613,:]),axis=1)
X = np.concatenate((X, blip_tags_img_text[:305613,:]),axis=1)
X = np.concatenate((X, blip_title_img_img[:305613,:]),axis=1)
X = np.concatenate((X, blip_title_img_mutimodal[:305613,:]),axis=1)
X = np.concatenate((X, blip_title_img_text[:305613,:]),axis=1)

print('X shape:', X.shape)
Y = df_label.values
Y = np.reshape(Y,[-1])
Y2=np.asarray(Y,np.int)


X2 = df_test.values

X2=X2[:,:ss_len]
X2= np.concatenate((X2, imf[305613:,:]),axis=1)
X2= np.concatenate((X2, roberta2[305613:,:]),axis=1)
X2= np.concatenate((X2, roberta1[305613:,:]),axis=1)
X2 = np.concatenate((X2, tfidf_cap[305613:,:]),axis=1)
X2 = np.concatenate((X2, cat1[305613:,:]),axis=1)
X2 = np.concatenate((X2, cat2[305613:,:]),axis=1)
X2 = np.concatenate((X2, cat3[305613:,:]),axis=1)
X2 = np.concatenate((X2, user_description[305613:,:]),axis=1) 
X2 = np.concatenate((X2, multi_lang_emb_tag[305613:,:]),axis=1)
X2 = np.concatenate((X2, multi_lang_emb_title[305613:,:]),axis=1)
X2 = np.concatenate((X2, cap_emb[305613:,:]),axis=1)
X2 = np.concatenate((X2, img_emb_eva[305613:,:]),axis=1)

X2 = np.concatenate((X2, blip_tags_img_img[305613:,:]),axis=1)
X2 = np.concatenate((X2, blip_tags_img_mutimodal[305613:,:]),axis=1)
X2 = np.concatenate((X2, blip_tags_img_text[305613:,:]),axis=1)
X2 = np.concatenate((X2, blip_title_img_img[305613:,:]),axis=1)
X2 = np.concatenate((X2, blip_title_img_mutimodal[305613:,:]),axis=1)
X2 = np.concatenate((X2, blip_title_img_text[305613:,:]),axis=1)

print('Test data shape:', X2.shape)
# del roberta1, roberta2, imf

oof_preds = np.zeros(X.shape[0])
feature_importance_df = pd.DataFrame()
t_value = np.zeros(X2.shape[0])
t_cnt = 0



tabnet_params = dict(
    n_d=32,  
    n_a=32, 
    n_steps=10, 
    gamma=1.3,  
    lambda_sparse=1e-4,  
    optimizer_fn=torch.optim.Adam,  
    optimizer_params=dict(lr=2e-2, weight_decay=1e-4), 
    mask_type="entmax", 
    scheduler_params=dict(
        mode="min",
        patience=5,
        min_lr=1e-5,
        factor=0.9
    ),  
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,  
    seed=2020
)


bsts = []
t_value = np.zeros((X2.shape[0]))

t_values = np.zeros((X2.shape[0], 5))
t_value_median = np.zeros((X2.shape[0]))

folds = [1,2,3,4,5]

list_Ytest =[]
list_pred =[]

stop_epochs =[]

start_time = time.time()

for fold in folds:# train_index, test_index , kf.split(X)
    train_index = df_train_fold.index[df_train_fold['Fold']!=fold]
    test_index = df_train_fold.index[df_train_fold['Fold']==fold]
    print("TRAIN:", train_index, "TEST:", test_index)
    print("TRAIN:", len(train_index), "TEST:", len(test_index))
    
    bst = TabNetRegressor(**tabnet_params)

    # 训练模型
    bst.fit(
                    X_train=X[train_index], y_train=Y[train_index].reshape(-1 ,1),
                    eval_set=[(X[test_index], Y[test_index].reshape(-1 ,1))],
                    patience=10, max_epochs=1000,
                    batch_size=1024, virtual_batch_size=128,
                    num_workers=0, drop_last=False,
                    eval_metric=["mae", "mse"]
                )


    predictions = bst.predict(X[test_index])
    bsts.append(bst)
    
    list_Ytest.append(Y[test_index])
    list_pred.append(predictions)
    
    # stop_epochs.append(history['epoch'])
    
    # t_value += bst.predict(X2)
    
    t_value_median = np.expand_dims(bst.predict(X2), axis=1)
    t_values[:, fold-1] =  np.squeeze(t_value_median) #[:, 0]


for validY, pred, bst in zip(list_Ytest, list_pred, bsts):
    print('SRC: %.6f' % (stats.spearmanr(validY, pred)[0]), end=', ')
    print('MAE : %.6f, MSE : %.6f' % (mean_absolute_error(validY, pred), mean_squared_error(validY, pred)))

allYtest = np.concatenate(list_Ytest)
allpreds = np.concatenate(list_pred)
print('[CV]')
print('SRC: %.6f' % (stats.spearmanr(allYtest, allpreds)[0]), end=', ')
print('MAE : %.6f, MSE : %.6f' % (mean_absolute_error(allYtest, allpreds), mean_squared_error(allYtest, allpreds)))

     
t_value = np.mean(t_values, axis=1) #t_value/5
t_value_median = np.median(t_values, axis=1)

end_time = time.time()
total_time = end_time - start_time
print("5-fold executed time", total_time, "s")

#save mean result
order = t_value.argsort()
ranks = order.argsort()
ranks = max(ranks) - ranks+1
df_cat = pd.read_json("data_2022/test/test_category.json")

y=pd.DataFrame(t_value)
y.columns=['y']
df_res = df_cat.join(y)
df_res=df_res[['Pid','y']]
df_res=df_res.astype({'Pid':int})
df_res.columns=['post_id','popularity_score']
res_key = pd.DataFrame(['result' for i in range(len(df_res))])
df_res = df_res.join(res_key)

df_res.columns=['post_id','popularity_score', 'result']
df_res['post_id'] = df_res['post_id'].astype('str')

j = '{"version": "VERSION 1.2","result":['

for ind, (x,y) in enumerate(df_res[['post_id','popularity_score']].values):
    if ind == len(df_res)-1:
        j += '{"post_id": "%s","popularity_score": %f}' % (x,y)
    else:
        j += '{"post_id": "%s","popularity_score": %f},' % (x,y)
    
j+=']}'
with open('submission-tabnet.json','w') as fp:
    fp.write(j)

print('Good job!')



# ensemble


with open('submissions/submission-lgbm.json', 'r') as f:
    data_1 = json.load(f)
with open('submissions/submission-tabnet.json', 'r') as f:
    data_2 = json.load(f)
    

results_1 = pd.DataFrame(data_1['result'])
results_2 = pd.DataFrame(data_2['result'])

ensemble = results_1.copy()


j = '{"version": "VERSION 1.2","result":['

for ind, (x, y) in enumerate(ensemble[['post_id', 'popularity_score']].values):
    if ind == len(ensemble) - 1:
        j += '{"post_id": "%s","popularity_score": %f}' % (x, y)
    else:
        j += '{"post_id": "%s","popularity_score": %f},' % (x, y)

j += ']}'

with open('submission.json','w') as fp:   
    fp.write(j)

print('Good job!')



















































































