import warnings
warnings.filterwarnings(action='ignore')

import gc
import os
import math
import json
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataloader import *
from model_relative import *

def arg_parser():
    parser = argparse.ArgumentParser(description='Distribution Learning Parser')
    parser.add_argument('--adm3info', default='./data/Adm2_Adm3_all.csv', type=str, help='row by row adm1-adm2-adm3 information')
    parser.add_argument('--adm3album', default='./album_main', type=str, help='path for pictures of adm3')
    parser.add_argument('--baseline', default='./model/baseline_model.ckpt',type=str,help='path for baseline model')
    parser.add_argument('--thrpics_src', default="./threshold_bag.json",type=str,help='path for threshold pictures. Selected in a different procedure with training.')
    parser.add_argument('--alpha', default=1, type=int, help='1 for alpha 1, 2 for alpha 1/log_4_5, 3 for alpha 1/log_9_10')
    parser.add_argument('--epochs', default=20, type=int,help='total epochs')
    parser.add_argument('--lr', default=5e-5, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--seed', default=486, type=int, help='fixed random seed')
    parser.add_argument('--gpu_devices', default=[0,1,2,3], nargs='+', type=int, help='gpu machines to use. please check before using')
    parser.add_argument('--lambdaLfit', default=1, type=int,help='Lambda for L_fit')
    parser.add_argument('--lambdaLadm2', default=1, type=int,help='Lambda for L_adm2')
    parser.add_argument('--modelsavepath', default='./model/', type=str, help='path for saving output models')
    parser.add_argument('--trainadm3csv',default='./data/Adm2_Adm3_train.csv', type=str, help='For training: row by row adm1-adm2-adm3 information')
    parser.add_argument('--adm2popcsv',default='./data/Adm2_Popinfo.csv', type=str, help='path for adm2 pop')
    
    return parser.parse_args()

def getitembyname (args, adm2name, adm3name):
    single_data = SingleDataset(adm2name=adm2name, adm3name=adm3name, metadata = args.adm3info,
                            root_album = args.adm3album,
                            transform = transforms.Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    single_loader = torch.utils.data.DataLoader(single_data, batch_size=1, shuffle=False, num_workers=1)
    return list(enumerate(single_loader))[0][1]

def bring_threshold_pics (args):
    file_path = args.thrpics_src
    
    with open(file_path, "r") as json_file:
        threshold_bag = json.load(json_file)
    
    th1list = threshold_bag['threshold10']
    th2list = threshold_bag['threshold200']
    
    return th1list,th2list

def save_checkpoint(state, dirpath, model, arch_name):
    filename = '{}.ckpt'.format(arch_name)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)

def calculate_thrscores(model, th1list, th2list, device):
    model.eval()
    with torch.no_grad():
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        score1 = 0
        score2 = 0
        for thr1pic in th1list:
            thr1pic = Image.open(thr1pic)
            thr1pic = transform(thr1pic).unsqueeze(0).to(device)
            thr1temp = model.module.simpleforward(thr1pic)
            thr1temp = torch.clamp(thr1temp, min = 0, max = 1000000000)
            score1 += thr1temp
        for thr2pic in th2list:
            thr2pic = Image.open(thr2pic)
            thr2pic = transform(thr2pic).unsqueeze(0).to(device)
            thr2temp = model.module.simpleforward(thr2pic)
            thr2temp = torch.clamp(thr2temp, min = 0, max = 1000000000)
            score2 += thr2temp
        score1 = torch.clamp(score1/len(th1list), min = 0)
        score2 = torch.clamp(score2/len(th2list), min = score1)
    return (score1.item(), score2.item())

def give_distribution(graphpd):    
    if args.alpha == 1:
        alpha = 1
    elif args.alpha == 2:
        alpha = 1/math.log(5,4)
    elif args.alpha == 3:
        alpha = 1/math.log(10,9)
    else:
        raise Exception('Not valid alpha value')
        
    magicdist = []
    for i in range(7):
        magicdist.append(1/pow((i+1),alpha))
            
    if sum(magicdist) != 0.0:
        magicdist = list(map(lambda x:x/sum(magicdist),magicdist))
    else:
        magicdist = list(map(lambda x:x,magicdist))
    graphtype = 'Reciprocal'
    return (magicdist,graphtype)

def main_distribution(args,th1list,th2list,device):
    Lintraloss = nn.MSELoss()
    Linterloss = nn.MSELoss()
    
    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    model = BinMultitask_relative(net, feature_size)
    model.load_state_dict(torch.load(args.baseline)['state_dict'], strict=True)    
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids = args.gpu_devices)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    trainadm3pd = pd.read_csv(args.trainadm3csv,index_col=0)
    traincitylist = list(set(trainadm3pd['adm2'].tolist()))
    
    adm2poppd = pd.read_csv(args.adm2popcsv,index_col=0)
    for row in adm2poppd.itertuples():
        if row.adm2 in traincitylist:
            continue
        adm2poppd = adm2poppd.drop(row.Index)
    adm2poppd = adm2poppd.sort_values(by='pop',ascending=False)
    adm2poppd = adm2poppd.reset_index(drop=True)
    adm2poppd.index += 1
    
    for epoch in range(args.epochs):
        print('----------- epoch: {} -----------'.format(epoch+1))
        model.eval()
        giventh1, giventh2 = calculate_thrscores(model, th1list, th2list, device)
        
        mcities = list(set(trainadm3pd['adm2'].tolist()))
        mcities.sort()
        random.seed(epoch)
        mcities_train = random.sample(mcities,int(len(mcities) * 0.8))
        mcities_val = [item for item in mcities if item not in mcities_train]
        
        city_idx = 1
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        
        print("---------------------training step---------------------")
        for mcity in mcities_train:
            model.eval()
            giventh1,giventh2 = calculate_thrscores(model, th1list, th2list, device)
            
            model.train()
            model.module.freeze_bn()
            model.module.freeze_front()
            
            adm3list = trainadm3pd[trainadm3pd['adm2']==mcity]['adm3'].tolist()
            if len(adm3list) < 7:
                gc.collect()
                torch.cuda.empty_cache()
                print("Pass!")
                continue
            
            newdict = dict()
            fsumzero = torch.tensor(0.0).to(device)
            for adm3name in adm3list:
                sample = getitembyname(args, mcity, adm3name)
                sample['images'] = torch.autograd.Variable(sample['images'].squeeze().to(device))
                adm3imgnum = len(sample['images'])
                if sample['images'].dim() == 4:
                    _, tempb, templogits = model(sample['images'],giventh1,giventh2)
                else:
                    _, tempb, templogits = model(sample['images'].unsqueeze(0),giventh1,giventh2)
                mysum = torch.sum(tempb)
                fsumzero += mysum
                scorelist = tempb 
                newdict[adm3name] =(mysum,adm3imgnum,scorelist)
             
            # Calculate L_inter
            popzero = adm2poppd[adm2poppd['adm2']==mcity]['pop'].values[0]
            magicidx = adm2poppd[adm2poppd['adm2']==mcity].index - 1
            
            if magicidx.values[0] == 0:
                loss_big = torch.tensor(0.0).to(device)
            else:
                citybig = adm2poppd.iloc[magicidx-1]['adm2'].values[0] #1 rank higher
                guonelist = trainadm3pd[trainadm3pd['adm2']==citybig]['adm3'].tolist()
                fsumbig = torch.tensor(0.0).to(device)
                for guone in guonelist:
                    gusample = getitembyname(args, citybig, guone)
                    gusample['images'] = torch.autograd.Variable(gusample['images'].squeeze().to(device))
                    if gusample['images'].dim() == 4:
                        _, tempb, templogits = model(gusample['images'],giventh1,giventh2)
                    else:
                        _, tempb, templogits = model(gusample['images'].unsqueeze(0),giventh1,giventh2)
                    fsumbig += torch.sum(tempb)
                popbig = adm2poppd[adm2poppd['adm2']==citybig]['pop'].values[0]
                loss_big = Linterloss(torch.div(fsumzero,fsumbig).float(), torch.tensor(popzero/popbig).to(device).float())
            
            if magicidx.values[0] == len(adm2poppd) -1:
                loss_small = torch.tensor(0.0).to(device)
            else:
                citysmall = adm2poppd.iloc[magicidx+1]['adm2'].values[0] #1 rank lower
                gutwolist = trainadm3pd[trainadm3pd['adm2']==citysmall]['adm3'].tolist()
                fsumsmall = torch.tensor(0.0).to(device)
                for gutwo in gutwolist:
                    gusample = getitembyname(args, citysmall, gutwo)
                    gusample['images'] = torch.autograd.Variable(gusample['images'].squeeze().to(device))
                    if gusample['images'].dim() == 4:
                        _, tempb, templogits = model(gusample['images'],giventh1,giventh2)
                    else:
                        _, tempb, templogits = model(gusample['images'].unsqueeze(0),giventh1,giventh2)
                    fsumsmall += torch.sum(tempb)
                popsmall = adm2poppd[adm2poppd['adm2']==citysmall]['pop'].values[0]
                loss_small = Linterloss(torch.div(fsumsmall,fsumzero).float(), torch.tensor(popsmall/popzero).to(device).float())
            
            inter_loss = args.lambdaLadm2 * (loss_small + loss_big)
            
            # Calculate L_intra
            fsigmadist = torch.tensor([]).to(device)
            for tempkey in newdict.keys():
                fsigmadist = torch.cat([fsigmadist, torch.sum(newdict[tempkey][2]).unsqueeze(0).unsqueeze(0)],dim=1)    
            fsigmadist = torch.sort(fsigmadist,descending=True)[0][0][:7]

            graphpd = pd.DataFrame(columns=["sigmaf"],dtype=object)
            for sigmaf in fsigmadist.tolist():
                newdata = {'sigmaf':sigmaf}
                graphpd = graphpd.append(newdata,ignore_index=True)
            graphpd.index = graphpd.index + 1
            mcitydist, mcitytype = give_distribution(graphpd)

            if torch.sum(fsigmadist) == 0:
                print("fsigmadist sum 0 error")
                continue
            fsigmadist = torch.div(fsigmadist,torch.sum(fsigmadist))
            mcitydist = torch.tensor(mcitydist).to(device)
            intra_loss = args.lambdaLfit * Lintraloss(fsigmadist.float(),mcitydist.float())
            
            train_loss = inter_loss + intra_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epoch_train_loss = epoch_train_loss + train_loss
            
            if city_idx % 5 == 0:
                print('{} total loss: {}'.format(mcity,train_loss))
                print('L_inter Loss : {}'.format(inter_loss.item()))
                print('L_intra Loss : {}'.format(intra_loss.item()))
                print(giventh1, giventh2)
                print()
            
            #for GPU memory
            del sample, gusample
            gc.collect()
            torch.cuda.empty_cache()
            city_idx = city_idx + 1
        
        print("---------------------validation step---------------------")
        city_idx = 1
        
        model.eval()
        giventh1, giventh2 = calculate_thrscores(model, th1list, th2list, device)
        print('Validation Step Threshold: {}/{}'.format(giventh1, giventh2))
        with torch.no_grad():
            for mcity in mcities_val:
                adm3list = trainadm3pd[trainadm3pd['adm2']==mcity]['adm3'].tolist()
                if len(adm3list) < 7:
                    continue
                
                newdict = dict()
                fsumzero = torch.tensor(0.0).to(device)
                for adm3name in adm3list:
                    sample = getitembyname(args, mcity, adm3name)
                    sample['images'] = torch.autograd.Variable(sample['images'].squeeze().to(device))
                    adm3imgnum = len(sample['images'])
                    if sample['images'].dim() == 4:
                        _, tempb, templogits = model(sample['images'],giventh1,giventh2)
                    else:
                        _, tempb, templogits = model(sample['images'].unsqueeze(0),giventh1,giventh2)
                    mysum = torch.sum(tempb)
                    fsumzero += mysum
                    scorelist = tempb 
                    newdict[adm3name] =(mysum,adm3imgnum,scorelist)

                # Calculate L_inter
                popzero = adm2poppd[adm2poppd['adm2']==mcity]['pop'].values[0]
                magicidx = adm2poppd[adm2poppd['adm2']==mcity].index - 1

                if magicidx.values[0] == 0:
                    loss_big = torch.tensor(0.0).to(device)
                else:
                    citybig = adm2poppd.iloc[magicidx-1]['adm2'].values[0] #1 rank higher
                    guonelist = trainadm3pd[trainadm3pd['adm2']==citybig]['adm3'].tolist()
                    fsumbig = torch.tensor(0.0).to(device)
                    for guone in guonelist:
                        gusample = getitembyname(args, citybig, guone)
                        gusample['images'] = torch.autograd.Variable(gusample['images'].squeeze().to(device))
                        if gusample['images'].dim() == 4:
                            _, tempb, templogits = model(gusample['images'],giventh1,giventh2)
                        else:
                            _, tempb, templogits = model(gusample['images'].unsqueeze(0),giventh1,giventh2)
                        fsumbig += torch.sum(tempb)
                    popbig = adm2poppd[adm2poppd['adm2']==citybig]['pop'].values[0]
                    loss_big = Linterloss(torch.div(fsumzero,fsumbig).float(), torch.tensor(popzero/popbig).to(device).float())

                if magicidx.values[0] == len(adm2poppd) -1:
                    loss_small = torch.tensor(0.0).to(device)
                else:
                    citysmall = adm2poppd.iloc[magicidx+1]['adm2'].values[0] #1 rank lower
                    gutwolist = trainadm3pd[trainadm3pd['adm2']==citysmall]['adm3'].tolist()
                    fsumsmall = torch.tensor(0.0).to(device)
                    for gutwo in gutwolist:
                        gusample = getitembyname(args, citysmall, gutwo)
                        gusample['images'] = torch.autograd.Variable(gusample['images'].squeeze().to(device))
                        if gusample['images'].dim() == 4:
                            _, tempb, templogits = model(gusample['images'],giventh1,giventh2)
                        else:
                            _, tempb, templogits = model(gusample['images'].unsqueeze(0),giventh1,giventh2)
                        fsumsmall += torch.sum(tempb)
                    popsmall = adm2poppd[adm2poppd['adm2']==citysmall]['pop'].values[0]
                    loss_small = Linterloss(torch.div(fsumsmall,fsumzero).float(), torch.tensor(popsmall/popzero).to(device).float())

                inter_loss = args.lambdaLadm2 * (loss_small + loss_big)

                # Calculate L_intra
                fsigmadist = torch.tensor([]).to(device)
                for tempkey in newdict.keys():
                    fsigmadist = torch.cat([fsigmadist, torch.sum(newdict[tempkey][2]).unsqueeze(0).unsqueeze(0)],dim=1)    
                fsigmadist = torch.sort(fsigmadist,descending=True)[0][0][:7]

                graphpd = pd.DataFrame(columns=["sigmaf"],dtype=object)
                for sigmaf in fsigmadist.tolist():
                    newdata = {'sigmaf':sigmaf}
                    graphpd = graphpd.append(newdata,ignore_index=True)
                graphpd.index = graphpd.index + 1
                mcitydist, mcitytype = give_distribution(graphpd)

                if torch.sum(fsigmadist) == 0:
                    print("fsigmadist sum 0 error")
                    continue
                fsigmadist = torch.div(fsigmadist,torch.sum(fsigmadist))
                mcitydist = torch.tensor(mcitydist).to(device)
                intra_loss = args.lambdaLfit * Lintraloss(fsigmadist.float(),mcitydist.float())
                
                val_loss = inter_loss + intra_loss
                epoch_val_loss = epoch_val_loss + val_loss
                
                if city_idx % 5 == 0:
                    print('{} total loss: {}'.format(mcity,val_loss))
                    print('L_inter Loss : {}'.format(inter_loss.item()))
                    print('L_intra Loss : {}'.format(intra_loss.item()))
                    print()
                
                #for GPU memory
                del sample,gusample
                gc.collect()
                torch.cuda.empty_cache()
                city_idx = city_idx + 1
        
        save_checkpoint({'state_dict': model.state_dict()}, args.modelsavepath, model, 'Epoch_'+str(epoch+1)+'_trainloss_'+str(round(epoch_train_loss.item(),4))+'_valloss_'+str(round(epoch_val_loss.item(),4)))
        print('------- epoch {}, train loss: {} / val loss: {} -------'.format(epoch+1, round(epoch_train_loss.item(),4),round(epoch_val_loss.item(),4)))
        

if __name__ == '__main__':
    args = arg_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    th1list, th2list = bring_threshold_pics(args)
    main_distribution(args,th1list,th2list, device)
    print("All done!")
