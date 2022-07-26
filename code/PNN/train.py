'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-18 18:18:23
 # @ Description:
 '''



import torch
from torch import nn
from model import PNN
from utils import create_criteo_dataset
from torch.utils.data import DataLoader
import argparse
import sklearn

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="args")
    parser.add_argument('-k', type=int, help="hidden dimension of k", default=8)
    parser.add_argument('-lr', type=float, help='学习率', default=1e-2)
    parser.add_argument('-epoch', type=int, help='eopch', default=100)
    args = parser.parse_args()

    data_path = '../code/Data/criteo.txt'
    feature_columns, train_data, test_data = create_criteo_dataset(data_path, split=0.2)
    train_dataloader = DataLoader(train_data, batch_size=16)
    test_dataloader = DataLoader(test_data, batch_size=16)

    k = args.k

    model = PNN(feature_columns, k)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)

    train_step = 0

    for i in range(args.epoch):
        # print("-----------epoch {}--------".format(i+1))
        sum_loss = []
        for data in train_dataloader:
            X, target = data[0], data[1]
            output = model(X)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_step += 1

            if train_step % 100 == 0:
                print("training step:{}, loss:{:.6f}".format(train_step, loss))

        if (i+1) % 30 == 0:
            total_test_loss = 0
            total_acc = 0
            with torch.no_grad():
                for data in test_dataloader:
                    X, target = data[0], data[1]
                    output = model(X)
                    loss = loss_fn(output, target)
                    total_test_loss += loss
                    output_ = torch.Tensor([1 if x > 0.5 else 0 for x in output]).reshape(-1,1)
                    total_acc += (output_ == target).sum()
            print("val in epoch {}, test loss: {:.5f},test acc: {:.5f}".format(i+1, total_test_loss, total_acc / len(test_data)))
            # torch.save(model, "model_{}.pth".format(i+1))





        







