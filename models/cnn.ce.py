import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import sklearn.metrics as sm
# from torchstat import stat
import torch.nn.functional as F
from torchsummary import summary

class AdaptiveReweight(nn.Module):
    def __init__(self, channel, reduction=4,momentum=0.1,index=0):
        self.channel=channel
        super(AdaptiveReweight, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LayerNorm([channel // reduction]),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.register_buffer('running_scale', torch.zeros(1))
        self.momentum=momentum
        self.ind=index
        

    def forward(self, x):
        b, c, _, _ = x.size()
        _x=x.view(b,c,-1)
        x_var=_x.var(dim=-1)

        y = self.fc(x_var).view(b, c)

        if self.training:
            scale=x_var.view(-1).mean(dim=-1).sqrt()
            self.running_scale.mul_(1. - self.momentum).add_(scale.data*self.momentum)
        else:
            scale=self.running_scale
        inv=(y/scale).view(b,c,1,1)
        return inv.expand_as(x)*x  
    
class CE(nn.Module):
    def __init__(self, num_features, pooling=False, num_groups=1, num_channels=64, T=3, dim=4, eps=1e-5, momentum=0,
                    *args, **kwargs):
        super(CE, self).__init__()
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.dim = dim

        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
            num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features

        self.AR=AdaptiveReweight(num_features)
        self.pool=None
        if pooling:
            self.pool=nn.MaxPool2d(2,stride=2)
    
        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))

        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        self.x_weight = nn.Parameter(torch.zeros(1))
        print(self.num_channels)

    def forward(self, X):
        N,C,H,W=X.size()
        xin=self.AR(X)
        x_pool=self.pool(X) if self.pool is not None else X
        
        x_pool=x_pool.transpose(0, 1).contiguous().view(self.num_groups, self.num_channels, -1)
        x = X.transpose(0, 1).contiguous().view(self.num_groups, self.num_channels, -1)
        _, d, m = x.size()
        
        if self.training:
            mean = x_pool.mean(-1, keepdim=True)
            
            xc = x_pool - mean
            
            P = [None] * (self.T + 1)
            P[0] = torch.eye(d,device=X.device).expand(self.num_groups, d, d)
            Sigma = torch.baddbmm(alpha=self.eps, input=P[0], beta=1. / m, batch1=xc, batch2=xc.transpose(1, 2))

            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            Sigma_N = Sigma * rTr
            for k in range(self.T):
                mat_power3=torch.matmul(torch.matmul(P[k],P[k]),P[k])
                P[k + 1] = torch.baddbmm(alpha=1.5, input=P[k], beta=-0.5, batch1=mat_power3, batch2=Sigma_N)
            
            wm = P[self.T]  

            self.running_mean.mul_(1. - self.momentum).add_(mean.data*self.momentum)
            self.running_wm.mul_((1. - self.momentum)).add_(self.momentum * wm.data)
        else:
            xc = x - self.running_mean
            wm = self.running_wm

        xn = wm.matmul(x)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()

        x_weight=torch.sigmoid(self.x_weight)
        return x_weight*Xn+(1-x_weight)*xin

class CNN_UCI(nn.Module):
    def __init__(self):
        super(CNN_UCI, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
        )
        self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
        self.layer6 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
        )
        self.ce = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Sequential(
                nn.Linear(15360, 6)
            )

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.ce(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        # x = F.normalize(x.cuda())
        return x

class ResCNN_UCI(nn.Module):
    def __init__(self):
            super(ResCNN_UCI, self).__init__()
            self.Block1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(64),
            )
            self.Block2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(128),
            )
            self.Block3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(256),
                nn.Dropout(0.5),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                # nn.Dropout(0.5),
                nn.ReLU(True)
            )
            self.shortcut3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 1)),
                nn.BatchNorm2d(256),
                # nn.Dropout(0.5)
            )
            self.ce = CE(num_features=256, pooling=False, num_channels=256)
            self.fc = nn.Sequential(
                nn.Linear(15360, 6)
            )

    def forward(self, x):
        h1 = self.Block1(x)
        r = self.shortcut1(x)
        h1 = h1 + r
        h2 = self.Block2(h1)
        r = self.shortcut2(h1)
        h2 = h2 + r
        h3 = self.Block3(h2)
        r = self.shortcut3(h2)
        h3 = h3 + r
        h3 = self.ce(h3)
        x = h3.view(h3.size(0), -1)
        x = self.fc(x)
        # x = nn.LayerNorm(x.size())(x.cpu())
        # x = x.cuda()
        x = F.normalize(x.cuda())
        return x


class CNN_UNIMIB(nn.Module):
    def __init__(self):
            super(CNN_UNIMIB, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
            )
            self.layer6 = nn.Sequential(
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
            )
            self.ce = CE(num_features=384, pooling=False, num_channels=384)

            self.fc = nn.Sequential(
                nn.Linear(8448, 17)
            )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.ce(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        # x = F.normalize(x.cuda())
        return


class ResCNN_UNIMIB(nn.Module):
    def __init__(self):
            super(ResCNN_UNIMIB, self).__init__()
            self.Block1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
                nn.ReLU(True)
            )
            self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(128),
            )
            self.Block2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
                nn.ReLU(True)
            )
            self.shortcut2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
                nn.BatchNorm2d(256),
            )
            self.Block3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True),
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
                nn.ReLU(True)
            )
            self.shortcut3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 2), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(384),
            )

            self.ce = CE(num_features=384, pooling=False, num_channels=384)

            self.fc = nn.Sequential(
                nn.Linear(8448, 17)
            )

    def forward(self, x):
        h1 = self.Block1(x)
        r = self.shortcut1(x)
        h1 = h1 + r
        h2 = self.Block2(h1)
        r = self.shortcut2(h1)
        h2 = h2 + r
        h3 = self.Block3(h2)
        r = self.shortcut3(h2)
        h3 = h3 + r
        h3 = self.ce(h3)
        x = h3.view(h3.size(0), -1)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()
        # x = F.normalize(x.cuda())
        return x


def CECNN_choose(dataset = 'uci', res=False):
    if dataset == 'uci':
        
        if res == False:
            # print('1')
            model = CNN_UCI()
        else:
            model = ResCNN_UCI()
        return model

    if dataset == 'unimib':
        if res == False:
            model = CNN_UNIMIB()
        else:
            model = ResCNN_UNIMIB()
        return model

    else:
        return print('not exist this model')

        

        



def main():
    model = CECNN_choose(dataset = 'uci', res=False).cuda()
    input = torch.rand(64, 1, 128, 9).cuda()
    output = model(input)
    print(output)
    # summary(model, (1, 128, 9))
    
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))

if __name__ == '__main__':
    main()