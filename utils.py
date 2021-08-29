import math
import torch
from torch.optim.optimizer import Optimizer, required
from torch import nn
import torch.nn.functional as F


# BasicNet
class BasicNet1(nn.Module):

    def __init__(
        self, args, use_bias=False, init="kaiming", use_multi_fc=False, device=None
    ):
        super(BasicNet1, self).__init__()

        self.use_bias = use_bias
        self.init = init
        self.use_multi_fc = use_multi_fc
        self.args = args

        self.convnet = RPS_net_mlp()    
        
        self.classifier = None

        self.n_classes = 0
        self.device = device
        self.cuda()
        
    def forward(self, x):
        x = self.convnet(x)
        return x


# RPS Net Module
class RPS_net_mlp(nn.Module):

        def __init__(self):
            super(RPS_net_mlp, self).__init__()
            self.init()

        def init(self):
            """Initialize all parameters"""
            self.mlp1 = nn.Linear(784, 400)
            self.mlp2 = nn.Linear(400, 400)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(400, 10, bias=False)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):

            x = x.view(-1, 784)

            x = self.mlp1(x)
            x = F.relu(x)
            
            x = self.mlp2(x)
            x = F.relu(x)

            x = self.fc(x)
            
            return x


# Accuracy
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))


    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
