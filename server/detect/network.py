import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

import argparse
import sys
import os

sys.path.append("/workspace/DFLive")    # 把项目根目录添加到path变量（根据项目实际目录进行调整）

from server.detect.celeb_dataset import CelebDataset
from server.detect.timit_dataset import TimitDataset


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch network')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.95, metavar='M',
                    help='SGD momentum')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()


train_set = CelebDataset()
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)

test_set = TimitDataset()
test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, num_workers=6)

# model = torchvision.models.resnet50(pretrained=True).cuda()
model = torch.load('./data/resnet50-finetuned.pth')

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = torch.load('./data/optim.pth')

criterion = nn.CrossEntropyLoss()


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
    results = ""
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        for o, t in zip(output, target):
            results += str(float(o[1] / (o[0] + o[1]))) + ' ' + str(int(t)) + '\n'
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_set), 100. * correct / len(test_set)))
    with open('./data/results.txt', 'w') as f:
        f.write(results)


test()
'''
for i in range(1, args.epochs+1):
    train(i)
    torch.save(model, './data/resnet50-finetuned2.pth')
    torch.save(optimizer, './data/optim2.pth')
    test()
'''