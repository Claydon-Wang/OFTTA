import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import sklearn.metrics as sm
# from torchstat import stat
import torch.nn.functional as F
from torchsummary import summary

def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1):
    module = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=0),
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))
    return module

## Define our DeepConvLSTM class, subclassing nn.Module.
class DeepConvLSTM(nn.Module):

    def __init__(self, n_channels, n_classes, dataset=None, experiment='default', conv_kernels=128, kernel_size=5,
                 LSTM_units=128, model='DeepConvLSTM'):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = conv_bn_relu(in_planes=1, out_planes=32, kernel=(kernel_size, 1))
        self.conv2 = conv_bn_relu(in_planes=32, out_planes=64, kernel=(kernel_size, 1))
        self.conv3 = conv_bn_relu(in_planes=64, out_planes=conv_kernels, kernel=(kernel_size, 1))
        self.conv4 = conv_bn_relu(in_planes=conv_kernels, out_planes=conv_kernels, kernel=(kernel_size, 1))

        
        # # self.conv1 = nn.Conv2d(1, 32, (kernel_size, 1))
        # self.conv1 = conv_bn_relu(in_planes=1, out_planes=32, kernel=(kernel_size, 1))
        # self.conv2 = nn.Conv2d(32, 64, (kernel_size, 1))
        # self.conv3 = nn.Conv2d(64, conv_kernels, (kernel_size, 1))
        # self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)

        self.classifier = nn.Linear(LSTM_units, n_classes)

        self.activation = nn.ReLU()

        self.model = model
        self.dataset = dataset
        self.experiment = experiment

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)

        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        feature = x
        out = self.classifier(x)

        return out, feature

def main():
    model = DeepConvLSTM(n_channels=9, n_classes=6).cuda()
    input = torch.rand(1, 1, 128, 9).cuda()
    output = model(input)
    print(output.shape)
    # summary(model, (1, 128, 9))
    
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))

def DeepConvLSTM_choose(dataset):
    if dataset == 'uci':      
        model = DeepConvLSTM(n_channels=9, n_classes=6, dataset='uci')
        return model

    elif dataset == 'unimib':
        model = DeepConvLSTM(n_channels=3, n_classes=17, dataset='unimib') 
        return model
    elif dataset == 'oppo':
        model = DeepConvLSTM(n_channels=77, n_classes=17, dataset='oppo') 
        return model


    else:
        return print('not exist this model')



if __name__ == '__main__':
    main()