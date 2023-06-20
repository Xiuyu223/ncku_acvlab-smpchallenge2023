# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:10:25 2023

@author: user
"""

import pandas as pd

df_user_data = pd.read_csv('test/test_userdata.txt', index_col=False)

location_description_len = df_user_data['location_description'][0].split(',')
print("数字个数为：", len(location_description_len))


df_user_data['location_description'] = df_user_data['location_description'].replace('\n', '0.0,'*399+'0.0+\r\n')
df_user_data['location_description'] = df_user_data['location_description'].str.strip()


user_description_len = df_user_data['user_description'][0].split(',')
print("数字个数为：", len(user_description_len))
zeros_list_399 = []


for i in range(399):
    zeros_str = '0.0' 
    zeros_list_399.append(zeros_str)
zeros_list_399 = ','.join(zeros_list_399)
user_description_NONE_COUNT=df_user_data.apply(lambda x: x.eq(zeros_list_399).sum())[2]
print(f'user_description !!!!!{user_description_NONE_COUNT} !!!!!equal to 1*399 0 vector')

df_user_data['user_description'] = df_user_data['user_description'].replace(zeros_list_399, pd.np.nan)
grouped = df_user_data.groupby('Uid')

filled = grouped['user_description'].fillna(method='ffill')
df_user_data['user_description'] = filled.where(filled.notnull(), df_user_data['user_description'])
df_user_data['user_description'] = df_user_data['user_description'].replace(pd.np.nan, zeros_list_399)

new_df = df_user_data.loc[:, ['Uid','Pid','canbuypro', 'ispro', 'location_description', 'photo_count', 'photo_firstdate', 'photo_firstdatetaken', 'timezone_offset', 'timezone_timezone_id', 'user_description']]

file_path = 'train/user_data_test_processed.pkl' 

try:
    df_user_data.to_pickle(file_path)
    print("DataFrame已成功保存為pickle文件。")
except IOError:
    print("保存文件時出錯。")
