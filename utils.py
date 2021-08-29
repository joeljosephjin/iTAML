import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.init()

    def init(self):
        """Initialize all parameters"""
        self.linear1 = nn.Linear(784, 400)
        self.linear2 = nn.Linear(400, 400)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(400, 10, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, params=None):
        out = x.view(-1, 784)
        if not params:
            out = self.linear1(out)
            out = F.relu(out)
            out = self.linear2(out)
            out = F.relu(out)
            out = self.linear3(out)
        else:
            out = F.linear(out, params['linear1.weight'], params['linear1.bias'])
            out = F.relu(out)
            out = F.linear(out, params['linear2.weight'], params['linear2.bias'])
            out = F.relu(out)
            out = F.linear(out, params['linear3.weight'])

        return out

    def cloned_state_dict(self):
        adapted_params = OrderedDict()
        for key, val in self.named_parameters():
            adapted_params[key] = val
        return adapted_params






class LearnerUtils():
    def __init__(self):
        pass
    
    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.args.lr *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.lr

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
