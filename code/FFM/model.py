'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-18 00:15:01
 # @ Description:
 '''

import torch
import numpy as np


class FFM(torch.nn.Module):
    def __init__(self, feature_columns, k) -> None:
        super().__init__()

        self.k = k

        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # 每个特征域的特征数
        self.field_features = [1 for _ in range(len(self.dense_feature_columns))] + [field['field_feat_num'] for field in self.sparse_feature_columns]
        # 总的特征数量 onehot之后的
        self.feature_num = sum(self.field_features)
        # 特征域的数量
        self.field_num = len(self.field_features)
        # 为每一个域构建embedding
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(self.feature_num, self.k) for _ in range(self.field_num)
        ])
        # 计算每个field onehot特征数的偏移
        self.offsets = np.array((0, *np.cumsum(self.field_features)[:-1]), dtype=np.long)
        # 初始化
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

        # 线性层 通过embedding实现
        self.fc = torch.nn.Embedding(self.feature_num, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))
    
    def forward(self, x):
        # 没有对连续特征离散化，不知道对不对
        # 根据偏移找到正确的feature索引  x.shape  [batch_size, field_num]
        x_c = x.clone().detach().long()
        x_c[:, :len(self.dense_feature_columns)] = 0
        x_f = x_c +  + x_c.new_tensor(self.offsets).unsqueeze(0)
        # 每个特征对于每个域的embedding  x_embeddings.shape [field_num, batch_szie, field_num, k]
        x_embeddings = [self.embeddings[i](x_f) for i in range(self.field_num)]
        # 因为公式中要和特征值相乘，因此这里进行保留，离散特征用1
        x_m = torch.cat((x[:, :len(self.dense_feature_columns)], torch.Tensor(torch.ones(x.shape[0], len(self.sparse_feature_columns)))), dim=1)
        cross_part = 0
        for i in range(self.field_num - 1):
            for j in range(self.field_num):
                cross_part += torch.sum(x_embeddings[i][:, j] * x_embeddings[j][:, i], dim=-1, keepdim=True) * (x_m[:, i] * x_m[:, j]).unsqueeze(1)

        linear = torch.sum(self.fc(x_f), dim=1) + self.bias

        return torch.sigmoid(linear + cross_part)
