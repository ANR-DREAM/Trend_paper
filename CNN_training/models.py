import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
