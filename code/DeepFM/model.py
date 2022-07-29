'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-29 19:03:19
 # @ Description:
 '''

import torch
import torch.nn as nn
from utils import *
from layers import Linear, FM_layer, DNN

class DeepFM(nn.Module):
    """
        DeepFM
    """
    def __init__(self, feature_columns) -> None:
        super().__init__()

        self.feature_columns = feature_columns
        self.sparse_embedding_matrix = create_embedding_layers(feature_columns)
        self.feature_index = create_input_feature_index(feature_columns)
        self.linear_model = Linear(self.feature_columns, self.feature_index)
        self.fm = FM_layer()
        self.dnn = DNN(compute_dnn_input_dims(self.feature_columns))

    def forward(self, X):
        # 构建input层
        dense_value_list, sparse_embedding_list = create_input_layers(X, self.feature_columns, self.sparse_embedding_matrix, self.feature_index)

        linear_logit = self.linear_model(X)

        # sparse_embedding_list 长度为fields的list 每个元素是[batch_size, dims] -> [batch_size, fields, dims]
        # FM 二阶交叉项
        fm_input = torch.stack(sparse_embedding_list, dim=1)
        fm_logit = self.fm(fm_input)

        dnn_input = create_dnn_input(dense_value_list, sparse_embedding_list)
        dnn_logit = self.dnn(dnn_input)

        output = torch.sigmoid(linear_logit + fm_logit + dnn_logit)
        return output



        



