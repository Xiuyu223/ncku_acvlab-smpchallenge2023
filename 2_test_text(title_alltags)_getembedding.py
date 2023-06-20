'''
To run this code, make sure sentence_transformers(https://github.com/UKPLab/sentence-transformers) installed.
'''
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

df = pd.read_csv('test/captioning_done_geo_category_none_test_processed_na_time.csv')
print('read!')


selected_column = df["Title"].tolist()
embeddings = embedder.encode(selected_column)
print(embeddings)
print(len(embeddings[0]))
embeddings = np.array(embeddings).reshape(-1, 1)
result_df = pd.DataFrame(embeddings, columns=["Title"])
result_df.to_pickle("test/Title.pkl")
print('title')


selected_column = df["Alltags"].tolist()
embeddings = model.encode(selected_column)
print(embeddings)
print(len(embeddings[0]))
embeddings = np.array(embeddings).reshape(-1, 1)
result_df = pd.DataFrame(embeddings, columns=["Alltags"])
result_df.to_pickle("test/Alltags.pkl")
print('alltag')




df_Spatial_Temperal = df[['Uid', 'Pid', 'timezone_timezone_id', 'Geoaccuracy', 'location_description', 'Post_year',
'Post_month', 'Post_day', 'Post_hour', 'Post_min', 'Post_sec', 'Post_week1_7', 'PostWeek_day_1_365',
'PhotoFirstdate_year', 'PhotoFirstdate_month', 'PhotoFirstdate_day', 'PhotoFirstdate_hour',
'PhotoFirstdate_min', 'PhotoFirstdate_sec', 'PhotoFirstdate_week1_7', 'PhotoFirstdatetaken_year',
'PhotoFirstdatetaken_month', 'PhotoFirstdatetaken_day', 'PhotoFirstdatetaken_hour',
'PhotoFirstdatetaken_min', 'PhotoFirstdatetaken_sec', 'PhotoFirstdatetaken_week_1_7',
'PhotoFirstdatetaken_day_1_365']]
df_Spatial_Temperal.to_pickle('test/newtest_spatial_temperal.pkl')

'''
df_label = df[['Uid', 'Pid', 'label']]
df_label.to_pickle('train/newtest_label.pkl')
'''
df_category = df[['Uid', 'Pid', 'Category', 'Subcategory', 'Concept', 'Mediatype', 'Mediastatus',
'canbuypro', 'ispro', 'Ispublic', 'City', 'State', 'Country']]
df_category.to_pickle("test/newtest_category.pkl")


df_user_info = df[['Uid','Pid','user_description','photo_count','totalImages','follower',
'following','totalViews','totalTags','totalGeotagged','totalFaves','totalInGroup']]
df_user_info.to_pickle('test/newtest_user_info.pkl')