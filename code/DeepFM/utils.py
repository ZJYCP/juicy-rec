'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-29 19:03:02
 # @ Description:
 '''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple, OrderedDict


# 特征标记
SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])

def create_criteo_dataset(data_path, split=0.2, sparse_dim=8):
    data = pd.read_csv(data_path)

    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1,27)]
    # 数据预处理  填充 归一化 离散编码
    data[dense_features] = data[dense_features].fillna(0)
    data[sparse_features] = data[sparse_features].fillna('-1')

    data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    feature_columns = [[DenseFeat(feat, 1) for feat in dense_features]] + \
           [[SparseFeat(feat, data[feat].nunique(), embed_dim=sparse_dim) for feat in sparse_features]]

    train, test = train_test_split(data, test_size=split)

    train_data = CriteoDataset(train)
    test_data = CriteoDataset(test)
    return feature_columns, train_data, test_data


# 数据集类
class CriteoDataset(Dataset):
    def __init__(self, data):
        self.length = len(data)
        self.data = data.reset_index(drop=True)

    def __getitem__(self, index):
        X = torch.Tensor(self.data.iloc[index, 1:].values)
        y = torch.Tensor(self.data.iloc[index,:1].to_numpy().reshape(1,))
        return X, y

    def __len__(self):
        return self.length

def create_embedding_layers(feature_columns, is_linear=False):
    """
    为离散特征构建embedding层
    """
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))

    embedding_layers = torch.nn.ModuleList(
        {feature.name: torch.nn.Embedding(feature.vocabulary_size, feature.embedding_dim if not is_linear else 1) for feature in sparse_feature_columns}
    )
    for embedding in embedding_layers:
        torch.nn.init.xavier_uniform_(embedding.weight.data)

    return embedding_layers

def create_input_feature_index(feature_columns):
    features = OrderedDict()
    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
    return features

def create_input_layers(X, feature_columns, embedding_layers, feature_index):
    """
        构建输入层
    """
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
    # 离散特征embedding
    sparse_embedding_list = [embedding_layers[feat.name](X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) 
                                    for feat in sparse_feature_columns]
    # 连续特征直接取出                                    
    dense_value_list = [X[:, feature_index[feat.name][0]: feature_index[feat.name][1]] 
                                for feat in dense_feature_columns]

    return dense_value_list, sparse_embedding_list

def create_dnn_input(dense_value_list, sparse_embedding_list):
    """
        构建DNN网络的输入，连接离散特征和连续特征
    """
    sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
    return torch.cat([sparse_dnn_input, dense_dnn_input], dim=-1)

def compute_dnn_input_dims(feature_columns):
    """
        计算dnn网络的输入数据维度  连续特征加embedding后的离散特征维度
    """
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
    dense_input_dims = sum(feat.dimension for feat in dense_feature_columns)    
    sparse_input_dims = sum(feat.embedding_dim for feat in sparse_feature_columns)

    return dense_input_dims + sparse_input_dims
