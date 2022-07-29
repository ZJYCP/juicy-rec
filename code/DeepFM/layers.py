'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-29 22:05:25
 # @ Description:
 '''

import torch
import torch.nn as nn
from utils import SparseFeat, DenseFeat, create_embedding_layers, create_input_feature_index

class FM_layer(nn.Module):
    """
        FM 二阶交叉项
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs):
        # inputs [batch_size, fields, dim]
        # 优化后的公式为： 0.5 * 求和（和的平方-平方的和）  =>> B x 1
        square_of_sum = torch.pow(torch.sum(inputs, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(inputs * inputs, dim=1, keepdim=True)
        cross_term = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=2)
        return cross_term

class Linear(nn.Module):
    """
         FM一阶线性层
    """
    def __init__(self, feature_columns, feature_index) -> None:
        super().__init__()
        
        self.feature_index = feature_index # 特征索引
        self.sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
        self.dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))

        ## 通过创建Embedding实现离散特征的线性变化
        self.sparse_embeddings = create_embedding_layers(feature_columns, is_linear=True)
        ## 连续型特征的linear层
        self.dense_w = nn.Parameter(torch.Tensor((sum(dense_feat.dimension for dense_feat in self.dense_feature_columns), 1)))
        torch.nn.init.xavier_uniform_(self.dense_w.data)

        self.bias = torch.nn.Parameter(torch.zeros((1,),dtype=torch.float64))
    
    def forward(self, X):
        # 离散特征线性变化，list中的每个元素是为每个特征域的变化
        sparse_embedding_list = [self.sparse_embeddings[feat.name](X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) 
                                    for feat in self.sparse_feature_columns]
        # 直接取连续型特征
        dense_value_list = [X[:, self.feature_index[feat.name][0]: self.feature_index[feat.name][1]] 
                                for feat in self.dense_feature_columns]

        sparse_logit = torch.sum(torch.cat(sparse_embedding_list, dim=-1), dim=-1)
        dense_logit = torch.cat(dense_value_list, dim=-1).matmul(self.dense_w) + self.bias

        linear_logit = (sparse_logit + dense_logit).reshape(-1, 1)
        return linear_logit


class DNN(nn.Module):
    """
        深度网络部分
    """
    def __init__(self, input_dims) -> None:
        super().__init__()
        self.dnn_net = nn.Sequential([
            nn.Linear(input_dims, input_dims//2),
            nn.ReLU(),
            nn.Linear(input_dims//2, input_dims//4),
            nn.ReLU(),
            nn.Linear(input_dims//4, 1)
        ])
    
    def forward(self,inputs):
        output = self.dnn_net(inputs)
        return output
