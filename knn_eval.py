
import models_eval
import numpy as np
import torch
import argparse

import torchvision
import torchvision.transforms as tfs

from knn import kNN
from util import AverageMeter, setup_runtime

import models_eval


parser = argparse.ArgumentParser(description='PyTorch Implementation')
parser.add_argument('--arch', default='resnetv1_18', type=str, help='architecture')
parser.add_argument('--ncl', default=128, type=int, help='number of clusters')
parser.add_argument('--hc', default=1, type=int, help='number of heads')
parser.add_argument('--device', default="0", type=str, help='cuda device')

parser.add_argument('--id', default="whu", type=str, help='dataset')
parser.add_argument('--ck', default="0", type=str, help='models_saved')

# optimization
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--batch-size', default=128, type=int, metavar='BS', help='batch size')
parser.add_argument('--datadir', default='/raid/lql/data/rs/div/', type=str)

args = parser.parse_args()

setup_runtime(2, [args.device])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = tfs.Compose([
    tfs.Resize(256),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class DataSet(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""

    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)

dataset_test = DataSet(torchvision.datasets.ImageFolder(args.datadir + args.id + '/train', transform_test))
trainloader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=False)

testset = DataSet(torchvision.datasets.ImageFolder(args.datadir + args.id + '/val', transform_test))
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=False)

print('number of training pics is ', len(trainloader.dataset))
print('number of testing pics is ', len(testloader.dataset))

print('==> Building model..')  ##########################################
numc = [args.ncl] * args.hc
model = models_eval.__dict__[args.arch](num_classes=numc)

model.to(device)
print(model)

Pth = '/raid/lql/models_saved/' + args.ck
model.load_state_dict(torch.load(Pth))

knn_dim = 512
kNN(model, trainloader, testloader, K=10, sigma=0.1, dim=knn_dim)
kNN(model, trainloader, testloader, K=50, sigma=0.1, dim=knn_dim)
kNN(model, trainloader, testloader, K=100, sigma=0.1, dim=knn_dim)
kNN(model, trainloader, testloader, K=200, sigma=0.1, dim=knn_dim)

