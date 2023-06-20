import os
import requests
from bs4 import BeautifulSoup
import csv
import shutil
# 設定工作目錄
os.chdir('imgs/')

# 讀取 train_new_imgpath.txt 和 newtrain_additional.txt 檔案
f = open('train/train_new_imgpath.txt')
line = f.readline().replace('\n', '')

f2 = open('train/newtrain_additional__.txt')

# 創建並打開 CSV 檔案
csv_file = open('train/downloaded_images.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['UID', 'PID', 'img_filepath'])
i=0
# 下載圖片並記錄路徑到 CSV 檔案
try:
    while i<27346:
        line2 = f2.readline().replace('\n', '')
        Uid, Pid, _, _, _ = line2.split('\t')
    
        res = requests.get(line)
        soup = BeautifulSoup(res.text)
        for img in soup.select('img'):
            fname = img['src'].split('/')[-1]
            path = '/imgs/%s/%s' % (Uid, Pid)
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            res2 = requests.get('https:' + img['src'], stream=True)
            f1 = open(fname, 'wb')
            shutil.copyfileobj(res2.raw, f1)
            f1.close()
            del res2
            print([Uid, Pid, os.path.join(path, fname)],i)
            csv_writer.writerow([Uid, Pid, os.path.join(path, fname)])
        i+=1
        line = f.readline().replace('\n', '')
except:
    pass
# 關閉 CSV 檔案
csv_file.close()
import pandas as pd
aa=pd.read_csv('train/downloaded_images.csv').drop_duplicates()
aa.to_csv('train/downloaded_images_modified.csv',index=False)
# 關閉檔案
f.close()
f2.close()
