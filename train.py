
from __future__ import print_function

import sys
import os
import argparse
import time

import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as tfs
from tensorboardX import SummaryWriter

from util import AverageMeter, setup_runtime
from augment import Augment, Cutout


parser = argparse.ArgumentParser(description='PyTorch Implementation')
parser.add_argument('--arch', default='resnetv1_18', type=str, help='architecture')
parser.add_argument('--ncl', default=128, type=int, help='number of clusters')
parser.add_argument('--hc', default=1, type=int, help='number of heads')
parser.add_argument('--nopts', default=300, type=int, help='number of label creating')
parser.add_argument('--nk', default=4, type=int, help='number of workers')

parser.add_argument('--device', default="0", type=str, help='cuda device')
parser.add_argument('--id', default='whu', type=str, help='dataset')

# optimization
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--epochs', default=300, type=int, help='number of epochs to train')
parser.add_argument('--bs', default=128, type=int, metavar='BS', help='batch size')

# logging saving etc.
parser.add_argument('--datadir', default='/raid/lql/data/rs/div/', type=str)
parser.add_argument('--exp', default='./exp', type=str, help='experimentdir')
args = parser.parse_args()

setup_runtime(2, [args.device])
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DataSet(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""

    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)

transforms = tfs.Compose([
    tfs.Resize(256),
    tfs.RandomResizedCrop(size=224, scale=(0.2, 1.)),
    tfs.RandomHorizontalFlip(),
    Augment(10),
    tfs.ToTensor(),
    tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    Cutout(
        n_holes=1,
        length=16,
        random=True,
    )
])

traindata = DataSet(torchvision.datasets.ImageFolder(args.datadir + args.id + '/train', transforms))
def td():
    loader = torch.utils.data.DataLoader(
        traindata,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.nk,
        pin_memory=True,
        drop_last=False,
    )
    return loader


N = len(td().dataset)
indx = 0
nl = 10

print('numb of training pics is ', N)

ave = torch.zeros(args.ncl).cuda()
for i in range(args.ncl):
    ave[i] = i ** indx
ave = ave * (N / torch.sum(ave))
print(torch.std(ave))
a = torch.zeros(args.ncl)
a[0] = N
max_std = torch.std(a)
del a
print('max_std is ', max_std)


def self_label(onk):
    torch.zeros(N, dtype=torch.int64).to(device)
    lbs = torch.argmax(onk, 1)

    ftor = 1
    decay = 1
    decay_bound = 1e-15
    ct = torch.cat((torch.bincount(lbs), torch.zeros(args.ncl - torch.max(lbs + 1)).to(device)), 0)
    std = torch.std(ct)
    print('true std is: {:.5f}'.format(std))

    if std == 0:
        return lbs
    else:
        ct = ct - ave
        ct = ct / torch.max(ct)
        acct = 0
        # k = 0
        t_opt = time.time()
        while decay > decay_bound:

            rate = std / max_std
            decay = rate * (torch.max(onk) - torch.min(onk)) / ftor

            am = (ct * decay).to(device)
            onk -= am
            lbs = torch.argmax(onk, 1)

            ct_new = torch.cat((torch.bincount(lbs), torch.zeros(args.ncl - torch.max(lbs + 1)).to(device)), 0)
            std_new = torch.std(ct_new)

            if std_new < std:
                acct = 0
                std = std_new
                ct = ct_new - ave
                ct = ct / torch.max(ct)
            elif std_new > std:
                acct = 0
                ftor = ftor * 1.5
                onk += am
            else:
                acct += 1
                ct = ct_new - ave
                ct = ct / torch.max(ct)
            if acct == 10 or std == 0:
                break
    print('opt cost:  ', time.time() - t_opt)
    print('target std is: {:.5f}'.format(std))
    print('  ')
    return lbs


def ps_lbs(model, td0, td1, td2, td3, td4, td5, td6, td7, td8, td9):
    monk = torch.zeros(nl, N, args.ncl).to(device)
    for d0, d1, d2, d3, d4, d5, d6, d7, d8, d9 in zip(td0, td1, td2, td3, td4, td5, td6, td7, td8, td9):
        data = {}
        idx = {}
        dt = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]
        for i in range(nl):
            data[i], idx[i] = dt[i][0].to(device), dt[i][2].to(device)
            monk[i][idx[i], :] = model(data[i]).detach()
    selflabels = torch.zeros(nl, N, dtype=torch.int64).to(device)
    for i in range(nl):
        selflabels[i] = self_label(monk[i])
    return selflabels


print('==> Building model..')  ##########################################
numc = [args.ncl] * args.hc
model = models.__dict__[args.arch](num_classes=numc)
model.to(device)


optimize_times = ((args.epochs + 1.0001) * N * (np.linspace(0, 1, args.nopts))[::-1]).tolist()
optimize_times = [(args.epochs + 10) * N] + optimize_times
print('We will optimize L at epochs:', [np.round(1.0 * t / N, 2) for t in optimize_times], flush=True)

selflabels = torch.zeros(nl, N, dtype=torch.int64).to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()


name = "%s" % args.exp.replace('/', '_')
writer = SummaryWriter(f'./runs/args.id')
writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch <= 160:
        lr = args.lr
    elif epoch <= 240:
        lr = args.lr * 0.1
    elif epoch <= args.epochs:
        lr = args.lr * 0.02
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch, selflabels, td0, td1, td2, td3, td4, td5, td6, td7, td8, td9):
    print('Epoch:  ', (epoch))
    print('Dateset : ', args.id)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    model.train()

    end = time.time()
    batch_idx = 0
    for t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 in zip(td0, td1, td2, td3, td4, td5, td6, td7, td8, td9):
        niter = epoch * len(td0) + batch_idx
        if niter * td0.batch_size >= optimize_times[-1]:
            with torch.no_grad():
                _ = optimize_times.pop()
                selflabels = ps_lbs(model, td(), td(), td(), td(), td(), td(), td(), td(), td(), td())
        data_time.update(time.time() - end)
        dt = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]
        inputs = {}
        indexes = {}
        outputs = {}
        for i in range(nl):
            inputs[i], indexes[i] = dt[i][0].to(device), dt[i][2].to(device)
            outputs[i] = model(inputs[i])

        optimizer.zero_grad()

        for i in range(nl):
            for j in range(nl):
                if i==0 and j==0:
                    loss = criterion(outputs[i], selflabels[j][indexes[i]])
                else:
                    loss += criterion(outputs[i], selflabels[j][indexes[i]])

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                epoch, batch_idx, len(td0), batch_time=batch_time, data_time=data_time,
                train_loss=train_loss))
            writer.add_scalar("loss", loss.item(), batch_idx * 512 + epoch * len(td0.dataset))
        batch_idx += 1
    return selflabels

# Pth = '/raid/lql/models_saved/pat_div_mt.pth'
# model.load_state_dict(torch.load(Pth))

PATH = '/raid/lql/models_saved/' + args.id + '_div_mt.pth'
print('       ')
print(args.id + ' pth = ',  PATH)
print('       ')

acc_best = 0.0
for epoch in range(args.epochs):
    t_ep = time.time()
    selflabels = train(epoch, selflabels, td(), td(), td(), td(), td(), td(), td(), td(), td(), td())
    torch.save(model.state_dict(), PATH)
    print(time.time()-t_ep)
    print('   ')
