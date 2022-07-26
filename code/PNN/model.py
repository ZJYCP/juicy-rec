'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-18 18:18:23
 # @ Description:
 '''



from tkinter import HIDDEN
from turtle import forward
from unittest import result
import torch
import numpy as np

class DNN(torch.nn.Module):
    def __init__(self, hideen_units, out_units) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(hideen_units, hideen_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hideen_units, out_units),
        )
    
    def forward(self, x):
        result = self.network(x)
        return result

class InnerProduction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        """
        params: x embedded [batch, field, k]
        """
        field_num = x.shape[1]
        row, col = [], []
        for i in range(field_num - 1):
            for j in range(i+1, field_num):
                row.append(i)
                col.append(j)
        # [batch_size, pair, k]
        q = torch.stack([x[:, idx, :] for idx in row], dim=1)
        p = torch.stack([x[:, idx, :] for idx in col], dim=1)
        inner_product = q * p

        inner_reduce = torch.sum(inner_product, dim=-1)

        return inner_reduce



class PNN(torch.nn.Module):
    def __init__(self, feature_columns, k, mode="inner") -> None:
        super().__init__()

        self.k = k
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.sparse_dims = len(self.sparse_feature_columns) * self.k
        self.sparse_field = len(self.sparse_feature_columns)

        self.mode = mode

        # 为每个离散特征构建embedding
        self.sparse_embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sparse_feature['field_feat_num'], self.k) for sparse_feature in self.sparse_feature_columns
        ])
        # 初始化
        for embedding in self.sparse_embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

        self.inner_product = InnerProduction()

        self.dense_dims = self.sparse_field * (self.sparse_field - 1) / 2 + self.sparse_dims + len(self.dense_feature_columns)
        self.dnn = DNN(int(self.dense_dims), 1)
    
    def forward(self, x):

        x_dense = x[:, :len(self.dense_feature_columns)]
        x_sparse = x[:, len(self.dense_feature_columns):]

        # shape [batch_size, filed_num, k]
        sparse_embed = torch.stack([emb_layer(x_sparse[:, i]) for i, emb_layer in enumerate(self.sparse_embeddings)]).permute(1, 0, 2)

        z = sparse_embed.reshape(-1, self.sparse_dims)

        if self.mode == "inner":
            inner_product = self.inner_product(sparse_embed)
            product_output = torch.cat((z, inner_product), dim=1)

        dnn_input = torch.cat((x_dense, product_output), dim=-1)

        dnn_output = self.dnn(dnn_input)

        output = torch.sigmoid(dnn_output)

        return output
