import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

import argparse
import sys

sys.path.append("/workspace/DFLive")

from server.detect.celeb_dataset import CelebDataset
from server.detect.timit_dataset import TimitDataset


parser = argparse.ArgumentParser(description='PyTorch network')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.95, metavar='M',
                    help='SGD momentum (default: 0.95)')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()


train_set = CelebDataset()
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)

test_set = TimitDataset()
test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, num_workers=6)

model = torchvision.models.resnet50(pretrained=True).cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_set),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_set), 100. * correct / len(test_set)))


# model = torch.load('./data/resnet50-finetuned.pth')
# test()
for i in range(1, args.epochs+1):
    train(i)
    torch.save(model, './data/resnet50-finetuned.pth')
    test()
