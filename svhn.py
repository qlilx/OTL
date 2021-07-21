#! /usr/bin/python

'''Train SVHN with PyTorch.
parts from  https://github.com/zhirongw/lemniscate.pytorch/blob/master/cifar.py,
https://github.com/yukimasano/self-label/blob/master/cifar.py, AET
'''
from __future__ import print_function

import sys
import os
import argparse
import time

import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as tfs
from tensorboardX import SummaryWriter

from util import AverageMeter
from svhn_utils import kNN, SVHNInstance, SVHNInstance_


def feature_return_switch(model, bool=True):
    """
    switch between network output or conv5features
        if True: changes switch s.t. forward pass returns post-conv5 features
        if False: changes switch s.t. forward will give full network output
    """
    if bool:
        model.headcount = 1
    else:
        model.headcount = args.hc
    model.return_features = bool

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ids = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

indx = 0
ncl = 128
nnp = 73257

ave = torch.zeros(ncl).cuda()
for i in range(ncl):
    ave[i] = i ** indx
ave = ave * (nnp / torch.sum(ave))
print(torch.std(ave))
a = torch.zeros(ncl)
a[0] = nnp
max_std = torch.std(a)
del a
print('max_std is ', max_std)


def self_label(onk):
    lbs = torch.zeros(nnp, dtype=torch.int32).to(device)
    lbs = torch.argmax(onk, 1)

    ftor = 1
    decay = 1
    decay_bound = 1e-15
    ct = torch.cat((torch.bincount(lbs), torch.zeros(ncl - torch.max(lbs + 1)).to(device)), 0)
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

            ct_new = torch.cat((torch.bincount(lbs), torch.zeros(ncl - torch.max(lbs + 1)).to(device)), 0)
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


def ps_lbs(model, tl):
    onk = torch.zeros(nnp, ncl).to(device)
    t_nk = time.time()
    for batch_idx, (data, _, idx) in enumerate(tl):
        # print(batch_idx)
        data = data.to(device)
        onk[idx, :] = model(data).detach()
    print('nk costs:   ', time.time() - t_nk)
    selflabels = self_label(onk)
    del onk
    torch.cuda.empty_cache()
    return selflabels


parser = argparse.ArgumentParser(description='PyTorch Implementation of LCT for SVHN')

# model
parser.add_argument('--arch', default='alexnet', type=str, help='architecture')
parser.add_argument('--ncl', default=ncl, type=int, help='number of clusters')
parser.add_argument('--hc', default=1, type=int, help='number of heads')

# number of producing pseudo labels
parser.add_argument('--nopts', default=400, type=int, help='number of producing pseudo labels')

# optimization
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.0, type=float, help='sgd momentum')
parser.add_argument('--epochs', default=400, type=int, help='number of epochs to train')
parser.add_argument('--bs', default=128, type=int, metavar='BS', help='batch size')

# data
parser.add_argument('--datadir', default='/raid/lql/data', type=str)
parser.add_argument('--exp', default='./svhn', type=str, help='experimentdir')
parser.add_argument('--type', default='0', type=int, help='svhn')

args = parser.parse_args()


# Data augmentation
print('==> Preparing data..')
transform_train = tfs.Compose([
    tfs.Resize(256),
    tfs.RandomResizedCrop(size=224, scale=(0.2, 1.)),
    tfs.ColorJitter(0.5, 0.5, 0.5, 0.5),
    tfs.RandomGrayscale(p=0.5),
    tfs.ToTensor(),
    tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = tfs.Compose([
    tfs.Resize(256),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = SVHNInstance(root=args.datadir, split='train', download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=16)
trainloader_ = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=16)
trainloader_1 = torch.utils.data.DataLoader(trainset, batch_size=args.bs*2, shuffle=True, num_workers=16)
trainloader_2 = torch.utils.data.DataLoader(trainset, batch_size=args.bs*4, shuffle=True, num_workers=16)

testset = SVHNInstance_(root=args.datadir, split='test', download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=16)

print('==> Building model...')  ##########################################
numc = [args.ncl] * args.hc
model = model.__dict__[args.arch](num_classes=numc)
knn_dim = 4096

N = len(trainloader.dataset)
optimize_times = ((args.epochs + 1.0001) * N * (np.linspace(0, 1, args.nopts))[::-1]).tolist()
optimize_times = [(args.epochs + 10) * N] + optimize_times
print('We will optimize L at epochs:', [np.round(1.0 * t / N, 2) for t in optimize_times], flush=True)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if args.epochs == 400:
        if epoch >= 160:
            lr = args.lr * (0.1 ** ((epoch - 160) // 80))  # i.e. 240,320
        print('learning rate: ',lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.epochs == 800:
        if epoch >= 320:
            lr = args.lr * (0.1 ** ((epoch - 320) // 160))  # i.e. 480, 640
        print('learning rate: ',lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.epochs == 1600:
        if epoch >= 640:
            lr = args.lr * (0.1 ** ((epoch - 640) // 320))
        print('learning rate: ',lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

name = "%s" % args.exp.replace('/', '_')
writer = SummaryWriter(f'./runs/svhn{args.type}/{name}')
writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)


selflabels = torch.zeros(N, dtype=torch.int32)
selflabels_ = selflabels
selflabels_1 = selflabels
selflabels_2 = selflabels
# Training
def train(epoch, selflabels, selflabels_, selflabels_1, selflabels_2, tl, tl_, tl_1, tl_2):
    print('\nEpoch: %d' % epoch)
    adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    batch_idx = 0
    for data, data_ in zip(tl, tl_):
        niter = epoch * len(tl) + batch_idx
        if niter * trainloader.batch_size >= optimize_times[-1]:
            with torch.no_grad():
                _ = optimize_times.pop()
                print('labels updadating')
                selflabels = ps_lbs(model, tl)
                selflabels_ = ps_lbs(model, tl_)
                selflabels_1 = ps_lbs(model, tl_1)
                selflabels_2 = ps_lbs(model, tl_2)
        data_time.update(time.time() - end)
        inputs, targets, indexes = data[0].to(device), data[1].to(device), data[2].to(device)
        inputs_, targets_, indexes_ = data_[0].to(device), data_[1].to(device), data_[2].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        outputs_ = model(inputs_)

        loss_1 = criterion(outputs, selflabels[indexes])
        loss_2 = criterion(outputs, selflabels_[indexes])
        loss_3 = criterion(outputs, selflabels_1[indexes])
        loss_4 = criterion(outputs, selflabels_2[indexes])

        loss_5 = criterion(outputs_, selflabels[indexes_])
        loss_6 = criterion(outputs_, selflabels_[indexes_])
        loss_7 = criterion(outputs_, selflabels_1[indexes_])
        loss_8 = criterion(outputs_, selflabels_2[indexes_])
        
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7 + loss_8

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
            writer.add_scalar("loss", loss.item(), batch_idx * 512 + epoch * len(trainloader.dataset))
        batch_idx += 1
    return selflabels, selflabels_, selflabels_1, selflabels_2


for epoch in range(args.epochs):
    t_ep = time.time()
    selflabels, selflabels_, selflabels_1, selflabels_2 = train(epoch, selflabels, selflabels_, selflabels_1, selflabels_2, trainloader, trainloader_, trainloader_1, trainloader_2)
    PATH = '/raid/lql/models_saved/svhn.pth'
    torch.save(model.state_dict(), PATH)
    print(time.time()-t_ep)

feature_return_switch(model, True)
kNN(model, trainloader, testloader, K=10, sigma=0.1, dim=knn_dim)
kNN(model, trainloader, testloader, K=50, sigma=0.1, dim=knn_dim)
kNN(model, trainloader, testloader, K=100, sigma=0.1, dim=knn_dim)
kNN(model, trainloader, testloader, K=200, sigma=0.1, dim=knn_dim)
