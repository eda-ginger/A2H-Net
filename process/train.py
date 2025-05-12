import sys
import torch

from config import *
# from utils.utils import *




def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        out = net(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train