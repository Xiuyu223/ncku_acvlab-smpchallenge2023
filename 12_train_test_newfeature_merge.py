import pandas as pd
pathalias = pd.read_csv('train/AllNewProfile_alli.txt')

train_origin = pd.read_csv('train/train_processed_na_time_merged.csv')
train_origin=train_origin.drop(columns=['Unnamed: 0'])
train_origin_path=pd.merge(train_origin,pathalias,on=['Pathalias'],how='left')



pathalias = pd.read_csv("train/AllNewProfile_alli.txt")

test_origin = pd.read_csv("test/test_processed_na_time_merged.csv",encoding='utf-8-sig')

test_origin=test_origin.drop(columns=['Unnamed: 0.2','Unnamed: 0.1','Unnamed: 0'])
test_origin_path=pd.merge(test_origin,pathalias,on=['Pathalias'],how='left')



train_origin_path.to_csv('train/geo_category_none_old_train_processed_na_time.csv',
                  index=False)


test_origin_path.to_csv('test/geo_category_none_test_processed_na_time.csv',
                  index=False)

