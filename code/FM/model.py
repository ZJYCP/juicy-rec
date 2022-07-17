'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-17 13:48:32
 # @ Description:
 '''
import torch


class FM(torch.nn.Module):
    def __init__(self, k, w_reg, v_reg, field_dims) -> None:
        super().__init__()

        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

        self.bias = torch.nn.Parameter(torch.zeros((1,),dtype=torch.float64))
        self.w = torch.nn.Parameter(torch.rand(field_dims, 1, dtype=torch.float64))
        self.v = torch.nn.Parameter(torch.rand(field_dims, self.k,dtype=torch.float64))

        torch.nn.init.xavier_uniform_(self.w.data)
        torch.nn.init.xavier_uniform_(self.v.data)
    
    def forward(self, x):
        linear = torch.matmul(x, self.w) + self.bias
        square_of_sum = torch.matmul(x, self.v) ** 2
        sum_of_square = torch.matmul(x**2, self.v**2)
        ix = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=1, keepdim=True)

        return torch.sigmoid(linear + ix)
