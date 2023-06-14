import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

class BinMultitask_relative(nn.Module):
    def __init__(self, backbone, feature_size):
        super(BinMultitask_relative, self).__init__()
        self.backbone = backbone
        self.W = nn.Parameter(torch.empty(feature_size, 1))
        nn.init.normal_(self.W)
        self.softmax = nn.Softmax(dim = 1)
        self.max = 1000000
        self.min = 0
        
    def forward(self, x, thr1score, thr2score):
        # Feature Extractor
        embed = self.backbone(x)        
        
        # Feature Extractor
        out = torch.matmul(embed, self.W)
        score = torch.clamp(out, min = self.min, max = self.max)
        
        dist1 = thr1score - score
        dist2 = thr2score - score
        
        # Logit Extractor for Ordinal Regression (urban / rural / uninhabited)
        logit = torch.cat((-dist2, torch.min(-dist1, dist2), dist1), dim = 1) 
        return embed, score, self.softmax(logit)
    
    def simpleforward(self, x):
        # Feature Extractor
        embed = self.backbone(x)        
        
        # Feature Extractor
        out = torch.matmul(embed, self.W)
        score = torch.clamp(out, min = self.min, max = self.max)
        
        return score
    
    def freeze_bn(self):
        '''Freeze BatchNorm mean/var.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
    def freeze_front(self):
        '''Freeze Except for 4th & last FC layer.'''
        for param in self.named_parameters():
            if (param[0] != 'W') and ('layer4' not in param[0]):
                param[1].requires_grad = False