import torch
import torch.nn as nn
import torch.nn.functional as F

########################################UNet
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels1, in_channels2, out_channels, bilinear=True,activation='relu'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels1 + in_channels2, out_channels,activation=activation)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
            self.conv = DoubleConv(2*in_channels2, out_channels,activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,rateDropout, mid_channels=None,activation='relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        if activation == 'relu':
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode='reflect'),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode='reflect'),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
        elif activation == 'tanh' :
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode='reflect'),
                    nn.BatchNorm2d(mid_channels),
                    nn.Tanh(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode='reflect'),
                    nn.BatchNorm2d(out_channels),
                    nn.Tanh(inplace=True) )
        elif activation == 'logsigmoid' :
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode='reflect'),
                    nn.BatchNorm2d(mid_channels),
                    nn.LogSigmoid(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode='reflect'),
                    nn.BatchNorm2d(out_channels),
                    nn.LogSigmoid(inplace=True) )
        elif activation == 'bilin' :
            self.double_conv = DoubleConvBILIN(in_channels, mid_channels,padding_mode='reflect')

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,rateDropout, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64,rateDropout)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

    
##############################################CNN
class CNN(nn.Module):
    """
    CNN simple
    """
    def __init__(self, n_channel,rateDropout):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channel,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,64,3, padding =(1,1))
        self.conv4 = nn.Conv2d(64,128,3, padding =(1,1))
        self.conv5 = nn.Conv2d(128,1,3, padding =(1,1))
        self.dropout = nn.Dropout(rateDropout)
        
    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.conv5(x)
        return x
    
##############################################CNN_W
# Definition of the W attention module for the multimode CNN :
class CNN_W(nn.Module):
    def __init__(self, n_channel,rateDropout):
        super(CNN_W, self).__init__()
        self.conv1 = nn.Conv2d(n_channel,16,3, padding =(1,1))    
        self.conv2 = nn.Conv2d(16,32,3, padding =(1,1))
        self.conv3 = nn.Conv2d(32,8,3, padding =(1,1))
        self.dropout = nn.Dropout(rateDropout)        

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = (F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = (F.softmax(self.conv3(x)))
        return x
