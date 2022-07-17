'''
 # @ Author: Chunpeng Yu
 # @ Create Time: 2022-07-17 13:48:27
 # @ Description: Implementation of FMs
 '''

from ast import arg
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model import FM
from utils import create_criteo_dataset
from torch.utils.data import DataLoader
import argparse
import sklearn

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="args")
    parser.add_argument('-k', type=int, help="hidden dimension of k", default=8)
    parser.add_argument('-w_reg', type=float, help='w正则', default=1e-4)
    parser.add_argument('-v_reg', type=float, help='v正则', default=1e-4)
    parser.add_argument('-lr', type=float, help='学习率', default=1e-2)
    parser.add_argument('-epoch', type=int, help='eopch', default=100)
    args = parser.parse_args()

    data_path = '../Data/criteo.txt'
    train_data, test_data = create_criteo_dataset(data_path, split=0.2)
    train_dataloader = DataLoader(train_data, batch_size=16)
    test_dataloader = DataLoader(test_data, batch_size=16)

    k = args.k
    w_reg = args.w_reg
    v_reg = args.v_reg
    field_dims = test_data.data.shape[1] - 1

    model = FM(k, w_reg, v_reg, field_dims)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)

    train_step = 0
    writer = SummaryWriter("log_train")

    for i in range(args.epoch):
        # print("-----------epoch {}--------".format(i+1))
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
                writer.add_scalar('train_loss', loss.item(), train_step)

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
            writer.add_scalar('test_loss', total_test_loss, (i+1)/10)
            writer.add_scalar('test_acc', total_acc / len(test_data), (i+1)/10)
            # torch.save(model, "model_{}.pth".format(i+1))

    writer.close()




        







