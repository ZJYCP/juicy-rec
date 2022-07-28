'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-28 14:12:18
 # @ Description:
 '''


import torch
import numpy as np

class Wide(torch.nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.rand(input_dim, 1, dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.w.data)

    def forward(self, x):
        output = torch.matmul(x, self.w)
        return output
        

class Deep(torch.nn.Module):
    def __init__(self, hideen_units, out_units) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(hideen_units, hideen_units//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hideen_units//2, hideen_units//4),
            torch.nn.ReLU(),
            torch.nn.Linear(hideen_units//4, out_units),
        )
    
    def forward(self, x):
        result = self.network(x)
        return result

class WideDeep(torch.nn.Module):
    def __init__(self, feature_columns, k, mode="inner") -> None:
        super().__init__()

        self.k = k
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # 离散特征域数量
        self.sparse_field = len(self.sparse_feature_columns)
        # 离散特征embedding之后的维度
        self.sparse_dims = self.sparse_field * self.k
        # 连续特征维度
        self.dense_field = len(self.dense_feature_columns)

        # 为每个离散特征构建embedding
        self.sparse_embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sparse_feature['field_feat_num'], self.k) for sparse_feature in self.sparse_feature_columns
        ])
        # 初始化
        for embedding in self.sparse_embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
        
        self.one_hot_dim = sum([sparse_feature['field_feat_num'] for sparse_feature in self.sparse_feature_columns])
        self.wide_net = Wide(self.one_hot_dim + self.dense_field)
        self.deep_net = Deep(self.dense_field+self.sparse_dims, 1)
    
    def forward(self, x):
        x_dense = x[:, :self.dense_field]
        x_sparse = x[:, self.dense_field:self.dense_field+self.sparse_field]
        one_hot = x[:, self.dense_field+self.sparse_field:]

        # Wide部分
        wide_input = torch.cat((x_dense, one_hot), dim=-1)
        wide_output = self.wide_net(wide_input)

        # Deep部分
        sparse_embed = torch.stack([emb_layer(x_sparse[:, i].long()) for i, emb_layer in enumerate(self.sparse_embeddings)]).permute(1, 0, 2).reshape(-1, self.sparse_dims)
        deep_input = torch.cat((x_dense, sparse_embed), dim=-1)
        deep_output = self.deep_net(deep_input)

        output = torch.sigmoid(wide_output + deep_output)
        return output
