import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import DeformConv2d

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )
    
def inverted_residual(inp,hidden,oup,stride_bn,stride_dw):
    return nn.Sequential(
        conv_bn1X1(inp,hidden,stride_bn),
        conv_dw(hidden,oup,stride_dw)
    )
    
def defconv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        Deform_Conv_V1(inp, oup, 1, stride, padding=0),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def defconv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        Deform_Conv_V1(inp, inp, 3, stride, 1, groups=inp),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        Deform_Conv_V1(inp, oup, 1, 1, 0),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )
    
def def_inverted_residual(inp,hidden,oup,stride_bn,stride_dw):
    return nn.Sequential(
        defconv_bn1X1(inp,hidden,stride_bn),
        defconv_dw(hidden,oup,stride_dw)        
    )
    
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)
        self.dcn = Deform_Conv_V1(out_channel, out_channel)
    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        out = self.dcn(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


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
            inverted_residual(16,64,32,2,1),
            inverted_residual(32,64,32,1,1),
            inverted_residual(32,64,64,1,2),
            inverted_residual(64,128,64,1,1)

            # conv_bn(3, 8, 2, leaky = 0.1),    # 3
            # conv_dw(8, 16, 1),   # 7
            # conv_dw(16, 32, 2),  # 11
            # conv_dw(192, 32, 1),  # 19
            # conv_dw(32, 64, 2),  # 27
            # conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            inverted_residual(64,128,128,1,2),
            inverted_residual(128,256,128,1,1),
            inverted_residual(128,256,128,1,1),
            inverted_residual(128,256,128,1,1),
            inverted_residual(128,256,128,1,1),
            inverted_residual(128,256,128,1,1),
            inverted_residual(128,256,128,1,1),
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
            # def_inverted_residual(128,512,256,1,2),
            # def_inverted_residual(256,512,256,1,1),
            inverted_residual(128,256,256,1,2),
            inverted_residual(256,256,256,1,1),
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
    
class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4):
        super(Inception,self).__init__()
        #线路1 1*1的卷积层
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c,c1,kernel_size=1),
            nn.ReLU()
        )
        #线路2 1*1卷积层后接3*3的卷积
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c,c2[0],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3,padding=1),
            nn.ReLU()
        )
        #线路3 1*1卷积层后接5*5的卷积层
        self.p3 = nn.Sequential(
            nn.Conv2d(in_c, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5,padding=2),
            nn.ReLU()
        )
        #线路4 3*3最大池化后接1*1卷积层
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_c,c4,kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return torch.cat((p1,p2,p3,p4),dim=1)
    
class Deform_Conv_V1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, offset_group=1):
        super(Deform_Conv_V1, self).__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv_offset = nn.Conv2d(
            in_channels,
            offset_channels * offset_group,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation= dilation,
        )
        self.DCN_V1 = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size= kernel_size,
            stride= stride,
            padding= padding,
            dilation = dilation,
            groups = groups,
            bias = False
        )
    def forward(self, x):
        offset = self.conv_offset(x)
        return self.DCN_V1(x, offset=offset)

if __name__ == "__main__":
    model = def_inverted_residual(10, 8, 5, 1, 1)
    print(model)
    x = torch.rand(4,10,60,60)
    print(x.shape)
    x = model(x)
    print(x.shape)
    torch.save(model.state_dict(), './test.pth')