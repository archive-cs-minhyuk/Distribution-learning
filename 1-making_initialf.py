import warnings
warnings.filterwarnings(action='ignore')

import os
import torch
import argparse
import torch.nn as nn
import pandas as pd
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import *
from dataloader import *

def arg_parser():
    parser = argparse.ArgumentParser(description='Fine tuning Parser')
    parser.add_argument('--traincsv', default='./metadata/proxy_metadata_train.csv', type=str, help='train metadata path')
    parser.add_argument('--testcsv', default='./metadata/proxy_metadata_test.csv', type=str, help='test metadata path')
    parser.add_argument('--picroot', default='./album_1000/', type=str, help='proxy image path')
    parser.add_argument('--th1', default=10, type=int, help='threshold for uninhabited-rural')
    parser.add_argument('--th2', default=200, type=int, help='threshold for rural-urban')
    parser.add_argument('--lr', default=1e-4, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--epochs', default=100, type=int,help='total training epochs')
    parser.add_argument('--batchsize', default=50, type=int, help='batch size')
    parser.add_argument('--modelsavepath', default='./model/', type=str, help='path for saving output models')
    
    return parser.parse_args()

def save_checkpoint(state, dirpath, model, arch_name):
    filename = '{}.ckpt'.format(arch_name)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)

def train_ordinal(train_loader, model, optimizer, epoch):
    model.train()
    count = 0                                                       
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):   
        inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())
        _, _, logit = model(inputs)
        # Soft Label Cross Entropy Loss
        loss = torch.mean(torch.sum(-targets * torch.log(logit+(1e-15)), 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
        
    total_loss /= count
    print('[Epoch: %d] loss: %.5f' % (epoch + 1, total_loss))
    
def test_ordinal(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    acc = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())
            _, _, logit = model(inputs)
            _, predicted = torch.max(logit, 1)
            _, answer =  torch.max(targets, 1)
            total += inputs.size(0)
            correct += (predicted == answer).sum().item()
        acc = (correct / total) * 100.0
        print('Test Acc : %.2f' % (acc))
    
    return acc

def main_pretrain(args):
    train_proxy = OproxyDataset(metadata = args.traincsv, 
                                root_dir = args.picroot,
                                transform=transforms.Compose([RandomRotate(),ToTensor(),Grayscale(prob = 0.1),
                                                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    test_proxy = OproxyDataset(metadata = args.testcsv, 
                                root_dir = args.picroot,
                                transform=transforms.Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    train_loader = torch.utils.data.DataLoader(train_proxy, batch_size=args.batchsize, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_proxy, batch_size=args.batchsize, shuffle=False, num_workers=4)
    
    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, args.th1, args.th2, ordinal=False)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    best_acc = 0
    for epoch in range(args.epochs):
        train_ordinal(train_loader, model, optimizer, epoch)
        if (epoch + 1) % 10 == 0:
            acc = test_ordinal(test_loader, model)
            if acc > best_acc:
                print('state_saving...')
                save_checkpoint({'state_dict': model.state_dict()}, args.modelsavepath, model, 'initialf_'+str(epoch+1))
                best_acc = acc

if __name__ == '__main__':
    args = arg_parser()
    main_pretrain(args)
    print("All done!")
