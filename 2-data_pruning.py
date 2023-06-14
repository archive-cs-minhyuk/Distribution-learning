import warnings
warnings.filterwarnings(action='ignore')

import os
import glob
import torch
import argparse
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from model import *
from dataloader import *

def arg_parser():
    parser = argparse.ArgumentParser(description='Fine tuning Parser')
    parser.add_argument('--targetalbum', default='./album_main', type=str, help='album path for pruning')
    parser.add_argument('--baseline', default='./model/baseline_model.ckpt',type=str,help='path for baseline model')
    parser.add_argument('--th1', default=10, type=int, help='threshold for uninhabited-rural')
    parser.add_argument('--th2', default=200, type=int, help='threshold for rural-urban')
    
    return parser.parse_args()

def main_prune (args):
    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, args.th1, args.th2, ordinal=False)  
    
    model.load_state_dict(torch.load(args.baseline)['state_dict'], strict=True)    
    model.cuda()
    print("Model loaded!")
    
    model.eval()
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    adm2pathlist = glob.glob(os.path.join(args.targetalbum,'*'))
    adm2pathlist.sort()
    
    total = 0
    for adm2path in adm2pathlist:
        count = 0
        adm3pathlist = glob.glob(os.path.join(adm2path,'*'))
        for adm3path in adm3pathlist:
            file_list = glob.glob(adm3path+'/*.png')
            for file in file_list:
                image = Image.open(file)
                imagetrans = transform(image).unsqueeze(0).cuda()
                _, score, _ = model(imagetrans)
                score = score.squeeze().item()
                if score < 10.0:
                    os.remove(file)
                    count += 1
        total += count
        print("Dir - {} : {} remove".format(adm2path.split('/')[-1], count))
    print("Total : {} remove".format(total))
    
    adm2pathlist = glob.glob(os.path.join(args.targetalbum,'*'))
    adm2pathlist.sort()

    for adm2path in adm2pathlist:
        adm3pathlist = glob.glob(os.path.join(adm2path,'*'))
        for adm3path in adm3pathlist:
            file_list = glob.glob(adm3path+'/*.png')
            if len(file_list) == 0:
                os.rmdir(adm3path)
                print(adm3path)

if __name__ == '__main__':
    args = arg_parser()
    main_prune(args)
    print("All done!")
