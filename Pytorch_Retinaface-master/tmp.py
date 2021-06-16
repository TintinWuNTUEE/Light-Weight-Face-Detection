import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
print(models.mobilenet_v2())
# print(models.resnet50())

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        
        self.stage1 = nn.Sequential(
            # inverted_residual(3,18,8),   
            # inverted_residual(8,32,16),   
            # inverted_residual(16,96,32),  
            # inverted_residual(32,192,32),  
            # inverted_residual(32,192,64),  
            # conv_dw(64,64,1),  
            inverted_residual(3,8,16,2,1),
            inverted_residual(16,96,32,2,1),
            inverted_residual(32,192,32,1,1),
            inverted_residual(32,192,64,1,2),
            inverted_residual(64,384,64,1,1)

            # conv_bn(3, 8, 2, leaky = 0.1),    # 3
            # conv_dw(8, 16, 1),   # 7
            # conv_dw(16, 32, 2),  # 11
            # conv_dw(192, 32, 1),  # 19
            # conv_dw(32, 64, 2),  # 27
            # conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            inverted_residual(64,384,128,1,2),
            inverted_residual(128,768,128,1,1),
            inverted_residual(128,768,128,1,1),
            inverted_residual(128,768,128,1,1),
            inverted_residual(128,768,128,1,1),
            inverted_residual(128,768,128,1,1),
            inverted_residual(128,768,128,1,1),
            # inverted_residual(128,768,128),
            # inverted_residual(128,768,128),
            # inverted_residual(128,768,128),        
            
            # conv_dw(64, 128, 2),  # 43 + 16 = 59

            # conv_dw(128, 128, 1), # 59 + 32 = 91

            # conv_dw(128, 128, 1), # 91 + 32 = 123
            # conv_dw(128, 128, 1), # 123 + 32 = 155
            # conv_dw(128, 128, 1), # 155 + 32 = 187
            # conv_dw(128, 128, 1), # 187 + 32 = 219
            # conv_dw(128, 128, 1),
        )
        self.stage3 = nn.Sequential(
            inverted_residual(128,768,256,1,2),
            inverted_residual(256,1536,256,1,1),
            # conv_dw(128, 256, 2), # 219 + 32 = 241
            # conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x