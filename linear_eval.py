import os
import argparse
import torch
import torchvision
import torchvision.transforms as tfs
import numpy as np
import torch.nn as nn
import torch.optim as optim

from util import AverageMeter, setup_runtime, py_softmax

import models_eval


def inference(loader, pre_model):
    feature_vector = []
    labels_vector = []
    for step, (x, y, _) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = pre_model(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(pre_model, train_loader, test_loader):
    train_X, train_y = inference(train_loader, pre_model)
    test_X, test_y = inference(test_loader, pre_model)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(loader, model, criterion, optimizer):
    # adjust_learning_rate(optimizer, epoch)
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def test(loader,  model, criterion):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def get_parser():
    parser = argparse.ArgumentParser(description='Driver')
    parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=120, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    parser.add_argument('--id', default='whu', type=str, help='dataset')
    parser.add_argument('--device', default="0", type=str, help='cuda device')
    parser.add_argument('--cl', default=12, type=int, help='classes')

    parser.add_argument('--arch', default='resnetv1_18', type=str, help='architecture')
    parser.add_argument('--ncl', default=128, type=int, help='number of clusters')
    parser.add_argument('--hc', default=1, type=int, help='number of heads')

    parser.add_argument('--datadir', default='/raid/lql/data/rs/div/', type=str)
    parser.add_argument('--ck', default='whu.pth', type=str)
    return parser


class DataSet(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""

    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_runtime(2, [args.device])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args)
    # Setup dataset

    transform_test = tfs.Compose([
        tfs.Resize(256),
        tfs.CenterCrop(224),
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = DataSet(torchvision.datasets.ImageFolder(args.datadir + args.id + '/train', transform_test))
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False)

    testset = DataSet(torchvision.datasets.ImageFolder(args.datadir + args.id + '/val', transform_test))
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False)

    print('==> Building model..')  ##########################################
    numc = [args.ncl] * args.hc
    pre_model = models_eval.__dict__[args.arch](num_classes=numc)

    Pth = '/raid/lql/models_saved/' + args.ck
    pre_model.load_state_dict(torch.load(Pth))
    pre_model.to(device)
    pre_model.eval()

    ## Logistic Regression
    n_features = 512
    n_classes = args.cl
    model = LogisticRegression(n_features, n_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
    criterion = torch.nn.CrossEntropyLoss()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        pre_model, train_loader, test_loader
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.batch_size
    )

    for epoch in range(args.epochs):
        loss_epoch, accuracy_epoch = train(
             arr_train_loader, model, criterion, optimizer
        )
        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
        )
        # testing
        if epoch % 1 == 0:
            loss_epoch, accuracy_epoch = test(
                arr_test_loader, model, criterion
            )
            print(
                f"[TEST]\t Loss: {loss_epoch / len(arr_test_loader)}\t -------------------TESTING Accuracy: {accuracy_epoch / len(arr_test_loader)}"
            )