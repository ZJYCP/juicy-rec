'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-18 18:18:23
 # @ Description:
 '''



from turtle import forward
import torch
import numpy as np


class Res_Layer(torch.nn.Module):
    def __init__(self, channel_in) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(channel_in, channel_in, dtype=torch.double)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        out = self.linear(x)
        res = self.relu(x + out)
        return res

class DeepCrossing(torch.nn.Module):
    def __init__(self, feature_columns, k) -> None:
        super().__init__()

        self.k = k
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.sparse_dims = len(self.sparse_feature_columns) * self.k

        # 为每个离散特征构建embedding
        self.sparse_embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sparse_feature['field_feat_num'], self.k) for sparse_feature in self.sparse_feature_columns
        ])
        # 初始化
        for embedding in self.sparse_embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

        self.dense_dims = self.sparse_dims + len(self.dense_feature_columns)
        self.res_layer = torch.nn.ModuleList([
            Res_Layer(self.dense_dims) for _ in range(2)
        ])

        self.out_layer = torch.nn.Linear(self.dense_dims, 1, dtype=torch.float64)
    
    def forward(self, x):

        x_dense = x[:, :len(self.dense_feature_columns)]
        x_sparse = x[:, len(self.dense_feature_columns):].long()
        sparse_embed = torch.stack([emb_layer(x_sparse[:, i]) for i, emb_layer in enumerate(self.sparse_embeddings)]).permute(1, 0, 2).reshape(-1, self.sparse_dims)

        feature_stack = torch.cat((x_dense, sparse_embed), dim=1)

        x = feature_stack
        for res_layer in self.res_layer:
            x = res_layer(x)

        score = self.out_layer(x)
        return torch.sigmoid(score)
