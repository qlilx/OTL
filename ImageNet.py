#! /usr/bin/python

'''Train ImageNet with PyTorch.
parts from  https://github.com/zhirongw/lemniscate.pytorch/blob/master/cifar.py,
https://github.com/yukimasano/self-label/blob/master/cifar.py,  AET
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

from tensorboardX import SummaryWriter

from util import AverageMeter


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


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
ids = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def self_label(onk, epoch):
    lbs = torch.zeros(nnp, dtype=torch.int32).to(device)
    lbs = torch.argmax(onk, 1)

    ftor = 1
    decay = 1
    decay_bound = 1e-15
    ct = torch.cat((torch.bincount(lbs), torch.zeros(ncl - torch.max(lbs + 1)).to(device)), 0)
    std = torch.std(ct)
    #raw_std[epoch+ep_td] = std
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
            if acct == 20 or std == 0:
                break
    opt_cost = time.time() - t_opt
    #opt_time[epoch+ep_td] = opt_cost
    print('opt cost:  ', opt_cost)
    #opt_std[epoch+ep_td] = std
    print('target std is: {:.5f}'.format(std))
    print('  ')
    # del onk
    # torch.cuda.empty_cache()
    return lbs


def ps_lbs(model, tl, epoch):
    onk = torch.zeros(nnp, ncl).to(device)
    t_nk = time.time()
    for batch_idx, (data, _, idx) in enumerate(tl):
        # print(batch_idx)
        data = data.to(device)
        onk[idx, :] = model(data).detach()
    print('nk costs:   ', time.time() - t_nk)
    selflabels = self_label(onk, epoch)
    # del onk
    # torch.cuda.empty_cache()
    return selflabels


parser = argparse.ArgumentParser(description='PyTorch Implementation of LCT for ImageNet')


# model
parser.add_argument('--arch', default='alexnet', type=str, help='architecture')
parser.add_argument('--ncl', default=3000, type=int, help='number of clusters')
parser.add_argument('--hc', default=1, type=int, help='number of heads')
parser.add_argument('--nopts', default=450, type=int, help='number of producing pseudo labels')

# optimization
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.0, type=float, help='sgd momentum')
parser.add_argument('--epochs', default=450, type=int, help='number of epochs to train')
parser.add_argument('--bs', default=256, type=int, metavar='BS', help='batch size')
parser.add_argument('--exp', default='./imagenet', type=str, help='experimentdir')
parser.add_argument('--type', default='0', type=int, help='imagenet')

# logging saving etc.
parser.add_argument('--datadir', default='/raid/lql/data/imagenet', type=str)
args = parser.parse_args()

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
print('==> Preparing data..')  #########################
from data import get_aug_dataloader
now = 16
trainloader = get_aug_dataloader(args.datadir, is_validation=False,
                                     batch_size=args.bs, image_size=256, crop_size=224,
                                     num_workers=now,
                                     augs=2, shuffle=True)
trainloader_ = get_aug_dataloader(args.datadir, is_validation=False,
                                     batch_size=args.bs, image_size=256, crop_size=224,
                                     num_workers=now,
                                     augs=2, shuffle=True)
trainloader_1 = get_aug_dataloader(args.datadir, is_validation=False,
                                     batch_size=384, image_size=256, crop_size=224,
                                     num_workers=now,
                                     augs=2, shuffle=True)
trainloader_2 = get_aug_dataloader(args.datadir, is_validation=False,
                                     batch_size=512, image_size=256, crop_size=224,
                                     num_workers=now,
                                     augs=2, shuffle=True)

print('==> Building model..')  ##########################################
numc = [args.ncl] * args.hc
model = model.__dict__[args.arch](num_classes=numc)


N = len(trainloader.dataset)
optimize_times = ((args.epochs + 1.0001) * N * (np.linspace(0, 1, args.nopts))[::-1]).tolist()
optimize_times = [(args.epochs + 10) * N] + optimize_times
print('We will optimize L at epochs:', [np.round(1.0 * t / N, 2) for t in optimize_times], flush=True)

# initialize pseudo labels
selflabels = torch.zeros(N, dtype=torch.int32).cuda()
selflabels_ = selflabels
selflabels_1 = selflabels
selflabels_2 = selflabels


model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)

name = "%s" % args.exp.replace('/', '_')
writer = SummaryWriter(f'./runs/imagenet{args.type}/{name}')
writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch <= 160 - ep_td:
        lr = args.lr
    elif epoch <= 300 - ep_td:
        lr = args.lr * 0.1
    elif epoch <= 380 - ep_td:
        lr = args.lr * 0.01
    else:
        lr = args.lr * 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch, selflabels, selflabels_, selflabels_1, selflabels_2, tl, tl_, tl_1, tl_2):
    print('Epoch:  ', (epoch + ep_td))
    print('IMAGENET')
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
                selflabels = ps_lbs(model, tl, epoch)
                selflabels_ = ps_lbs(model, tl_, epoch)
                selflabels_1 = ps_lbs(model, tl_1, epoch)
                selflabels_2 = ps_lbs(model, tl_2, epoch)
        data_time.update(time.time() - end)
        inputs, targets, indexes = data[0].to(device), data[1].to(device), data[2].to(device)
        inputs_, targets_, indexes_ = data_[0].to(device), data_[1].to(device), data_[2].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        outputs_ = model(inputs_)
        loss_1 = criterion(outputs_, selflabels[indexes_])
        loss_2 = criterion(outputs, selflabels_[indexes])
        loss_3 = criterion(outputs, selflabels[indexes])
        loss_4 = criterion(outputs_, selflabels_[indexes_])

        loss_5 = criterion(outputs, selflabels_1[indexes])
        loss_6 = criterion(outputs, selflabels_2[indexes])
        loss_7 = criterion(outputs_, selflabels_1[indexes_])
        loss_8 = criterion(outputs_, selflabels_2[indexes_])
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7 + loss_8

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 1000 == 0:
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                epoch+ep_td, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
            writer.add_scalar("loss", loss.item(), batch_idx * 512 + epoch * len(trainloader.dataset))
        batch_idx += 1
    return selflabels, selflabels_, selflabels_1, selflabels_2


N = len(trainloader.dataset)
indx = 0
ncl = 3000
nnp = N
print('number of pics is ', nnp)

ave = torch.zeros(ncl).to(device)
for i in range(ncl):
    ave[i] = i ** indx
ave = ave * (nnp / torch.sum(ave))
print(torch.std(ave))
a = torch.zeros(ncl)
a[0] = nnp
max_std = torch.std(a)
print(max_std)
del a


#opt_std = torch.zeros(450).to(device)
#opt_time = torch.zeros(450).to(device)
#raw_std = torch.zeros(450).to(device)

#opt_std = np.load('/raid/lql/npy_save/opt_std.npy')
#opt_time = np.load('/raid/lql/npy_save/opt_time.npy')
#raw_std = np.load('/raid/lql/npy_save/raw_std.npy')
#opt_std = torch.from_numpy(opt_std).to(device)
#opt_time = torch.from_numpy(opt_time).to(device)
#raw_std = torch.from_numpy(raw_std).to(device)

ep_td = 0


for epoch in range(450 - ep_td):
    t_ep = time.time()
    selflabels, selflabels_, selflabels_1, selflabels_2 = train(epoch, selflabels, selflabels_, selflabels_1,
                                                                selflabels_2, trainloader, trainloader_, trainloader_1,
                                                                trainloader_2)
    PATH = '/raid/lql/models_saved/imagenet.pth'
    torch.save(model.state_dict(), PATH)
    if epoch == 379 - ep_td:
        p379 = '/raid/lql/models_saved/imagenet_379ep.pth'
        torch.save(model.state_dict(), p379)
    #np.save('/raid/lql/npy_save/opt_std.npy', opt_std.cpu().numpy())
    #np.save('/raid/lql/npy_save/opt_time.npy', opt_time.cpu().numpy())
    #np.save('/raid/lql/npy_save/raw_std.npy', raw_std.cpu().numpy())
    print(time.time() - t_ep)