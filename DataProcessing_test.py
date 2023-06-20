import pandas as pd
from sklearn import metrics
import numpy as np
import pickle, pdb
import time, datetime
import gc
from scipy import stats
import json
from sklearn import feature_extraction

from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


cat_model = word2vec.Word2Vec.load('word2vec_cat.model')
tag_model = word2vec.Word2Vec.load('word2vec_tag.model')
cap_model = word2vec.Word2Vec.load('word2vec_cap.model')




def nor2(X,name):
    X.loc[:,name] = (X.loc[:,name]-np.mean(X.loc[:,name]))/(np.mean(X.loc[:,name])-np.std(X.loc[:,name]))
    return X

def listnor2(List):
    return (List-np.mean(List))/(np.mean(List)-np.std(List))

def nor(X,name):
    X.loc[:,name] = (X.loc[:,name]-np.min(X.loc[:,name]))/(np.max(X.loc[:,name])-np.min(X.loc[:,name]))
    return X

def listnor(List):
    return (List-np.min(List))/(np.max(List)-np.min(List))

def word_vec_list(list_w,m):
    model = m
    v = []
    for i in list_w:
        vec = np.zeros([model.vector_size,])
#         try:
#         import pdb
#         pdb.set_trace()
        vec = model.wv[i]
#         except:
#             pass
        v.append(vec)
    return pd.DataFrame(v)

def sen_to_vec(List,model):
    vec_of_s = []
    for i in List:
        s_v = []
        i = str(i)
        for j in i.split():
            w_v = np.zeros([model.vector_size,])
            try:
                w_v = model.wv[j]
            except:
                pass
            s_v.append(w_v)
        if s_v == []:
            s_v = np.zeros([model.vector_size,])
        else:
            s_v = sum(s_v)
        vec_of_s.append(s_v)
    
    return pd.DataFrame(vec_of_s)


def Convert_uniqueid(x, sets):
    return np.where(sets==x)[0][0]

def convert_imfeat(data):
    feat = []
    for i in range(len(data)):
        x=data[i].strip('\n')
        try:
            d = np.asarray([float(d) for d in x.split(' ')])
        except:
            d = np.zeros((1001))
        feat.append(d)
    feat = pd.DataFrame(feat)
    return feat

def createlist(ls_name,number):
    ls=list()
    for i in range(0,number):
        name=ls_name +'_'+ str(i+1)
        ls.append(name)
    return ls

def parsePathalias(x):
    if x == 'None':
        return 0
    else:
        return len(x)

def Convert_Date(x, ind):
    timestamp = x
    timeArray = time.localtime(timestamp)
    
    return timeArray[ind]

def Date2Ticks(x):
    Year='20'+x[-2:]
    Month=month[x[-6:-3]]
    Day=x[:-7]
    date1 = str(Year+'/'+Month+'/'+Day)
    return time.mktime(datetime.datetime.strptime(date1, "%Y/%m/%d").timetuple())

def Tfidf(text) :

    vec_of_s = []
    for i in text:
        i = str(i)
        vec_of_s.append(i)
    
    vectorizer = CountVectorizer()    
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(vec_of_s))
    
    word = vectorizer.get_feature_names() #所有文本的關鍵字
    weight = tfidf.toarray()              #對應的tfidf矩陣
    return weight


def PreProcessing_test():
    # load dataset
#     with open('train/train_tags.json', 'r', encoding='utf-8') as fp:
#         tags = json.loads(fp.read())
#     df_tags = pd.DataFrame.from_dict(tags, orient='columns')

    df_tags = pd.read_csv("train/train_tags.csv")
    df_add = pd.read_csv("train/train_additional.txt",delimiter=" ")
    df_cat = pd.read_csv("train/train_category.txt",delimiter=" ")
    df_ts = pd.read_json("train/train_temporalspatial.json")
    df_label = pd.read_csv("train/train_label.txt",delimiter=" ")
    df_cap = pd.read_csv("train/train_img_caption_only_blip.txt", delimiter=",", encoding='utf-8')#train_caption.txt
    df_imfeat = pd.read_csv('train/train_image_feat2.csv', delimiter=',')

    df_tags2 = pd.read_json("test/test_tags.json")
    df_add2 = pd.read_json("test/test_additional.json")
    df_cat2 = pd.read_json("test/test_category.json")
    df_ts2 = pd.read_json("test/test_temporalspatial.json")
    df_cap2 = pd.read_csv("test/test_img_caption_only_blip.txt",delimiter=",")#test_caption.txt
    df_imfeat2 = pd.read_csv('test/test_image_feat2.csv', delimiter=',')


    df_tags = df_tags.merge(df_tags2, how='outer')
    df_add = df_add.merge(df_add2, how='outer')
    df_cat = df_cat.merge(df_cat2, how='outer')
    df_ts = df_ts.merge(df_ts2, how='outer')
    df_cap = df_cap.merge(df_cap2, how='outer')
    df_imfeat=df_imfeat.merge(df_imfeat2, how='outer')
    
    '''
    Cat: Uid Pid Category Subcategory Concept
    TAG: Uid Pid Title Mediatype Alltags
    TS:  Uid Pid Postdate Latitude Longitude Geoaccuracy
    Add: Uid Pid Pathalias Ispublic Mediastatus

    '''

    cat1 = Tfidf(df_cat['Category'].values)
    cat2 = Tfidf(df_cat['Subcategory'].values)
    cat3 = Tfidf(df_cat['Concept'].values)
    
    
    tfidf_cap=Tfidf(df_cap['Caption'].values)
    
    
    enc_title=sen_to_vec(df_tags['Title'].values, cat_model) 
    enc_cap=sen_to_vec(df_cap['Caption'].values, cap_model)
    enc_cap.columns  = createlist('Caption',enc_cap.values.shape[1])
#     df_cap = df_cap.drop('Caption', axis=1)
#     enc_cap = listnor(enc_cap)
    df_cap=df_cap.join(enc_cap)
    
    dftag1=df_tags.Title.apply(lambda x: len(str(x)))
    dftag2=df_cap.Caption.apply(lambda x: len(x))
    dftag1.name = 'title_len'
    dftag2.name = 'caption_len'
    df_cap = df_cap.join(dftag2)
    df_tags = df_tags.join(dftag1)

    # date Conversion
    uidset =      np.unique(np.asarray(df_cat['Uid'].values.tolist()))
    pidset =      np.unique(np.asarray(df_cat['Pid'].values.tolist()))
    Mediatype =   np.unique(np.asarray(df_tags['Mediatype'].values.tolist()))
    print('Unique ID transfer done!')
    
    df_cat['Uid']=df_cat.Uid.apply(lambda x: Convert_uniqueid(x, uidset))
    df_cat['Pid']=df_cat.Pid.apply(lambda x: Convert_uniqueid(x, pidset))
    
    
    pd_cat1 = word_vec_list(df_cat['Category'].values, cat_model) 
    pd_cat2 = word_vec_list(df_cat['Subcategory'].values, cat_model) 
    pd_cat3 = word_vec_list(df_cat['Concept'].values, cat_model) 
    pd_cat1.columns  = createlist('cat',pd_cat1.values.shape[1])
    pd_cat2.columns  = createlist('subcat',pd_cat2.values.shape[1])
    pd_cat3.columns  = createlist('concept',pd_cat3.values.shape[1])
    
    c1set =      np.unique(np.asarray(df_cat['Category'].values.tolist()))
    c2set =      np.unique(np.asarray(df_cat['Subcategory'].values.tolist()))
    c3set =      np.unique(np.asarray(df_cat['Concept'].values.tolist()))
    dfcat1=df_cat.Category.apply(lambda x: len(x))
    dfcat2=df_cat.Subcategory.apply(lambda x: len(x))
    dfcat3=df_cat.Concept.apply(lambda x: len(x))
    
    df_cat['Category']=df_cat.Category.apply(lambda x: Convert_uniqueid(x, c1set))
    df_cat['Subcategory']=df_cat.Subcategory.apply(lambda x: Convert_uniqueid(x, c2set))
    df_cat['Concept']=df_cat.Concept.apply(lambda x: Convert_uniqueid(x, c3set))
    
    
    dfcat1.name = 'cat1_len'
    dfcat2.name = 'cat2_len'
    dfcat3.name = 'cat3_len'
    
    
#     df_cat = df_cat.drop(l['Category', 'Subcategory', 'Concept'], axis=1)
    df_cat = df_cat.join((pd_cat1))
    df_cat = df_cat.join((pd_cat2))
    df_cat = df_cat.join((pd_cat3))
    df_cat = df_cat.join(dfcat1)
    df_cat = df_cat.join(dfcat2)
    df_cat = df_cat.join(dfcat3)
    
    
    
    df_imfeat['Uid']=df_cat['Uid']
    df_imfeat['Pid']=df_cat['Pid']
    imfeat = convert_imfeat(df_imfeat['feat_1'])
    imfeat.columns  = createlist('imfeat',imfeat.values.shape[1])
    df_imfeat = df_imfeat.drop('feat_1', axis=1)
    df_imfeat = df_imfeat.join(imfeat)
    del imfeat
        
    df_cap['Uid']=df_cat['Uid']
    df_cap['Pid']=df_cat['Pid']
    

    # group data
    df_tags['Uid']=df_cat['Uid']
    df_tags['Pid']=df_cat['Pid']
    
    
    enc_tag= sen_to_vec(df_tags['Alltags'], tag_model) 
 
#     Mediatype   = pd.get_dummies(df_tags['Mediatype'])
#     Mediastatus = pd.get_dummies(df_add['Mediastatus'])
#     df_tags = df_tags.drop('Mediatype', axis=1)
#     df_tags = df_tags.join(Mediatype)
#     df_add = df_add.drop('Mediastatus', axis=1)
#     df_add = df_add.join(Mediastatus)
    
    enc_title.columns  = createlist('Title2',enc_title.values.shape[1])
    enc_tag.columns  = createlist('Alltags2',enc_tag.values.shape[1])
    df_tags = df_tags.drop(['Title', 'Alltags'], axis=1)
    df_tags=df_tags.join((enc_title))
    df_tags=df_tags.join((enc_tag))
    


    


    df_add['Uid']=df_cat['Uid']
    df_add['Pid']=df_cat['Pid']
    df_add['Ispublic']=df_add.Ispublic.apply(lambda x: int(x))
    enc_path = word_vec_list(df_add['Pathalias'], cat_model)
    enc_path.columns  = createlist('Pathalias2',enc_path.values.shape[1])
    df_add = df_add.drop('Pathalias', axis=1)
    df_add = df_add.join((enc_path))


    df_ts['Uid']=df_cat['Uid']
    df_ts['Pid']=df_cat['Pid']
    df_ts['Postyear']=df_ts.Postdate.apply(lambda x: Convert_Date(x, 0))
    df_ts['Postmonth']=df_ts.Postdate.apply(lambda x: Convert_Date(x, 1))
    df_ts['Postday']=df_ts.Postdate.apply(lambda x: Convert_Date(x, 2))
    df_ts['Posthour']=df_ts.Postdate.apply(lambda x: Convert_Date(x, 3))
    df_ts['Postmin']=df_ts.Postdate.apply(lambda x: Convert_Date(x, 4))
    df_ts['Postsec']=df_ts.Postdate.apply(lambda x: Convert_Date(x, 5))
    df_ts['Postweek']=df_ts.Postdate.apply(lambda x: Convert_Date(x, 6))
    df_ts['PostWyear']=df_ts.Postdate.apply(lambda x: Convert_Date(x, 7))
  
    df_ts['Latitude']=df_ts.Pid.apply(lambda x: float(x))
    df_ts['Longitude']=df_ts.Pid.apply(lambda x: float(x))
    
    df_ts = nor(df_ts,'Postdate')
    df_ts = nor2(df_ts,'Latitude')
    df_ts = nor2(df_ts,'Longitude')
    

    df_label['score'] = df_label.score.apply(lambda x: float(x))
    
    return df_tags, df_cat,df_add,df_ts, df_cap, df_imfeat,df_label, tfidf_cap, cat1, cat2, cat3
