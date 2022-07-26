'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-26 15:14:40
 # @ Description:
 '''


from numpy import float32
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'featname': feat, 'field_feat_num': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'featname': feat}

def create_criteo_dataset(data_path, split=0.2):
    data = pd.read_csv(data_path)

    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1,27)]

    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')

    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
    feature_columns = [[denseFeature(feat) for feat in dense_features]] + \
           [[sparseFeature(feat, data[feat].nunique(), embed_dim=8) for feat in sparse_features]]
    train, test = train_test_split(data, test_size=split)

    train_data = CriteoDataset(train)
    test_data = CriteoDataset(test)
    return feature_columns, train_data, test_data

class CriteoDataset(Dataset):
    def __init__(self, data):
        self.length = len(data)
        self.data = data.reset_index(drop=True)

    def __getitem__(self, index):
        X = torch.IntTensor(self.data.iloc[index, 1:].values)
        y = torch.Tensor(self.data.loc[index,'label'].reshape(1,))
        return X, y

    def __len__(self):
        return self.length


if __name__ == "__main__":
    train_data, test_data = create_criteo_dataset('../Data/criteo.txt')
    train_dataloader = DataLoader(train_data, batch_size=10)

