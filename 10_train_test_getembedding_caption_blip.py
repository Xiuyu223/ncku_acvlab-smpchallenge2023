from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import numpy as np
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


df = pd.read_csv('train/train_img_cap_list_blip.txt')
print('read!')
selected_column = df["Caption"].tolist()
embeddings = model.encode(selected_column)
print(embeddings)
print(len(embeddings[0]))
embeddings = np.array(embeddings).reshape(-1, 1)
result_df = pd.DataFrame(embeddings, columns=["Caption"])
print(len(result_df))
result_df.to_pickle("train/train_Caption_embedding_blip.pkl")
print('Caption_train')

df = pd.read_csv('test/test_img_cap_list_blip.txt')
print('read!')
selected_column = df["Caption"].tolist()
embeddings = model.encode(selected_column)
print(embeddings)
print(len(embeddings[0]))
embeddings = np.array(embeddings).reshape(-1, 1)
result_df = pd.DataFrame(embeddings, columns=["Caption"])
print(len(result_df))

result_df.to_pickle("test/test_Caption_embedding_blip.pkl")
print('Caption_test')