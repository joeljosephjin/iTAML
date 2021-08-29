import math
import torch
from torch.optim.optimizer import Optimizer, required
from torch import nn
import torch.nn.functional as F


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.convnet = RPS_net_mlp()    
        
    def forward(self, x):
        x = self.convnet(x)
        return x


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


class LearnerUtils():
    def __init__(self):
        pass
    
    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']

    def get_class_accs(self, pred, correct, class_acc, task_idx=None):
        for i,p in enumerate(pred.view(-1)):
            key = int(p.detach().cpu().numpy())

            if task_idx:
                key += self.args.class_per_task * task_idx

            if(correct[i]==1):
                if(key in class_acc.keys()):
                    class_acc[key] += 1
                else:
                    class_acc[key] = 1

        return class_acc

    def get_task_accuracies(self, class_acc):
        acc_task = {}
        for i in range(self.args.sess+1):
            acc_task[i] = 0
            for j in range(self.args.class_per_task):
                try:
                    acc_task[i] += class_acc[i*self.args.class_per_task+j]/self.args.sample_per_task_testing[i] * 100
                except:
                    pass
                    
        return acc_task
