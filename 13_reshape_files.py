import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# # user description train+test combine
# df_user_data_train = pd.read_pickle('train/user_data_train_processed.pkl')
# df_user_data_test = pd.read_pickle('test/user_data_test_processed.pkl')
# df_user_data_train = df_user_data_train.drop(['Unnamed: 0'], axis=1)
# df_user_data_test = df_user_data_test.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1)
# df_ts   = pd.read_pickle('data_2022/train/ts_processed_all.pkl')

# df_user_data = pd.concat([df_user_data_train, df_user_data_test], ignore_index=True)
# df_user_data = df_user_data.reset_index(drop=True)

# df_user_data = df_user_data.drop(['Category', 'Subcategory', 'Concept', 'Mediatype','Mediastatus','Ispublic', 'Title', 'Alltags', 'Pathalias',
#                                   'Geoaccuracy', 'Longitude', 'Latitude', 'Post_year', 'Post_month', 'Post_day', 'Post_hour', 'Post_min', 'Post_sec',
#                                   'img_filepath', 'label', 'pred'], axis=1)

# df_user_data.update({'Uid': df_ts['Uid'], 'Pid': df_ts['Pid']})
# df_user_data['Uid'] = df_user_data['Uid'].astype(int)

# # df_user_data.shape
# df_user_data.to_pickle('train/user_data_processed_all.pkl')
# del df_user_data_train, df_user_data_test, df_user_data

# df_user_data = pd.read_pickle('train/user_data_processed_all.pkl')

# split user_description to npy 
df_user_description = df_user_data['user_description'].str.split(',', expand=True)
new_columns = ['user_description_' + str(i+1) for i in range(df_user_description.shape[1])]
df_user_description.columns = new_columns

df_user_description = df_user_description.astype(float)

np.save('train/user_description.npy', df_user_description.to_numpy())

user_description = np.load('train/user_description.npy')
model=PCA(n_components=50)
model.fit(user_description)
user_description=model.transform(user_description)
np.save('train/user_description_pca50.npy', user_description)


# user_data_processed_all.pkl
df_ts   = pd.read_pickle('data_2022/train/ts_processed_all.pkl')
df = pd.read_pickle('train/user_data_processed_all.pkl')
df.update({'Uid': df_ts['Uid'], 'Pid': df_ts['Pid']})
df['Uid'] = df['Uid'].astype(int)
df.to_pickle('train/user_data_processed_all.pkl')


# newfeatures_processed_all.pkl
df_all = pd.read_pickle('data_2022/train/tags_processed_all.pkl')
df_new_features = df_all[['Uid', 'Pid']].copy()
df_1 = pd.read_csv('train/done_geo_category_none_old_train_processed_na_time.csv')
df_2 = pd.read_csv('test/done_geo_category_none_test_processed_na_time.csv')
df_1 = pd.concat([df_1, df_2], ignore_index=True)
new_features = df_1.drop(['Uid', 'Pid','photo_count', 'canbuypro', 'ispro', 'timezone_timezone_id', 'timezone_offset',
                        'Post_week1_7', 'PostWeek_day_1_365', 'PhotoFirstdate_year', 'PhotoFirstdate_month', 
                        'PhotoFirstdate_day', 'PhotoFirstdate_hour', 'PhotoFirstdate_min', 'PhotoFirstdate_sec', 
                        'PhotoFirstdate_week1_7', 'PhotoFirstdate_day_1_365', 'PhotoFirstdatetaken_year',
                        'PhotoFirstdatetaken_month', 'PhotoFirstdatetaken_day', 'PhotoFirstdatetaken_hour', 
                        'PhotoFirstdatetaken_min', 'PhotoFirstdatetaken_sec', 'PhotoFirstdatetaken_week_1_7',
                        'PhotoFirstdatetaken_day_1_365', 'img_filepath', 'label', 'img_filepath_local', 
                        'Ispublic', 'Pathalias', 'Geoaccuracy', 'Post_year', 'Post_month', 'Post_day', 'Post_hour', 'Post_min',
                        'Post_sec', 'Post_week1_7', 'PostWeek_day_1_365', 'Category', 'Subcategory', 'Concept',
                        'Mediastatus', 'Mediatype', 'user_description', 'Title', 'Alltags', 'location_description', 'pred'
                        ], axis = 1)
new_features.replace(-1.0, 0, inplace=True)
df_new_features = pd.concat([df_new_features, new_features], axis=1)
df_new_features['City'] = df_new_features['City'].fillna('NONE')
df_new_features['State'] = df_new_features['State'].fillna('NONE')
df_new_features['Country'] = df_new_features['Country'].fillna('NONE')

c1set = np.unique(np.asarray(df_new_features['City'].values.tolist()))
c2set = np.unique(np.asarray(df_new_features['State'].values.tolist()))
c3set = np.unique(np.asarray(df_new_features['Country'].values.tolist()))
c1set = np.append(c1set, 'NONE')
c2set = np.append(c2set, 'NONE')
c3set = np.append(c3set, 'NONE')

def Convert_uniqueid(x, sets):
    return np.where(sets==x)[0][0]

df_new_features['City']=df_new_features.City.apply(lambda x: Convert_uniqueid(x, c1set))
df_new_features['State']=df_new_features.State.apply(lambda x: Convert_uniqueid(x, c2set))
df_new_features['Country']=df_new_features.Country.apply(lambda x: Convert_uniqueid(x, c3set))

numeric_columns = ['totalImages', 'follower', 'following', 'totalViews', 'totalTags', 'totalGeotagged', 'totalFaves', 'totalInGroup']
df_new_features[numeric_columns] = df_new_features[numeric_columns].fillna(0)

df_new_features.to_pickle('train/newfeatures_processed_all.pkl')


# blip caption
df = pd.read_csv('train/train_img_cap_list_blip.txt', delimiter=',')
caption_txt = df['Caption']
caption_txt.to_csv('train/train_img_caption_only_blip.txt', index=None, header=None)

df = pd.read_csv('test/test_img_cap_list_blip.txt', delimiter=',')
caption_txt = df['Caption']
caption_txt.to_csv('test/test_img_caption_only_blip.txt', index=None, header=None)

# eva img embedding
df_1 = pd.read_pickle('train/train_img_embedding_eva.pkl')
df_2 = pd.read_pickle('test/test_img_embedding_eva.pkl')
df = pd.concat([df_1, df_2], ignore_index=True)
df = pd.DataFrame(df.apply(pd.Series))

npy = df[0].values
result = np.empty(npy.shape, dtype=object)
for i in range(len(npy)):
    result[i] = np.fromstring(npy[i][1:-1], dtype=float, sep=',')
    
img_emb_eva_all = np.stack([np.fromstring(x[1:-1], dtype=float, sep=',') for x in npy])

np.save('train/eva_img_embedding_all.npy', img_emb_eva_all)


# reorganize pkls
pkl='train_Title_embedding.pkl'
df = pd.read_pickle('train/' + pkl)
df = pd.DataFrame(df.values.reshape(305613,384))
df.to_pickle('train/' + pkl)

pkl='train_Alltags_embedding.pkl'
df = pd.read_pickle('train/' + pkl)
df = pd.DataFrame(df.values.reshape(305613,384))
df.to_pickle('ptrain/' + pkl)

pkl='test_Title_embedding.pkl'
df = pd.read_pickle('test/' + pkl)
df = pd.DataFrame(df.values.reshape(305613,384))
df.to_pickle('test/' + pkl)

pkl='test_Alltags_embedding.pkl'
df = pd.read_pickle('test/' + pkl)
df = pd.DataFrame(df.values.reshape(305613,384))
df.to_pickle('test/' + pkl)

n = 'Alltags'
train_data = pd.read_pickle('train/train_'+n+'_embedding.pkl')
test_data = pd.read_pickle('test/test_'+n+'_embedding.pkl')
combined_data = pd.concat([train_data, test_data], axis=0)
combined_data.to_pickle('train/'+n+'_embedding_all.pkl')

n = 'Title'
train_data = pd.read_pickle('train/train_'+n+'_embedding.pkl')
test_data = pd.read_pickle('test/test_'+n+'_embedding.pkl')
combined_data = pd.concat([train_data, test_data], axis=0)
combined_data.to_pickle('train/'+n+'_embedding_all.pkl')


# multi lang embedding (Alltags/Title, blip caption)
multi_lang_emb_tag = pd.read_pickle('train/Alltags_embedding_all.pkl').to_numpy()
model=PCA(n_components=20)
model.fit(multi_lang_emb_tag)
multi_lang_emb_tag=model.transform(multi_lang_emb_tag)
np.save('train/Alltags_embedding_all_pca20.npy', multi_lang_emb_tag)

multi_lang_emb_title = pd.read_pickle('train/Title_embedding_all.pkl').to_numpy()
model=PCA(n_components=20)
model.fit(multi_lang_emb_title)
multi_lang_emb_title=model.transform(multi_lang_emb_title)
np.save('train/Title_embedding_all_pca20.npy', multi_lang_emb_title)

cap_emb = pd.read_pickle('train/Caption_embedding_all.pkl').to_numpy()
model=PCA(n_components=20)
model.fit(cap_emb)
cap_emb=model.transform(cap_emb)
np.save('train/Caption_embedding_all_pca20.npy', cap_emb)


# img caption embedding blip
df = pd.read_pickle('train/train_Caption_embedding_blip.pkl')
df = pd.DataFrame(df.values[:117355392].reshape(305613,384))
df.to_pickle('train/train_Caption_embedding_blip.pkl')

df = pd.read_pickle('test/test_Caption_embedding_blip.pkl')
df = pd.DataFrame(df.values.reshape(180581,384))
df.to_pickle('test/test_Caption_embedding_blip.pkl')

train_data = pd.read_pickle('train/train_Caption_embedding_blip.pkl')
test_data = pd.read_pickle('test/test_Caption_embedding_blip.pkl')

df = pd.concat([train_data, test_data], ignore_index=True)
df.to_pickle('train/Caption_embedding_blip_all.pkl')








