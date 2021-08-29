import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from utils import *


class Learner(LearnerUtils):
    def __init__(self, model, args, trainloader, testloader):
        super(Learner, self).__init__()
        self.model=model
        self.args=args
        self.trainloader=trainloader 
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.testloader=testloader
        self.optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.001)

    def train(self):
        bi = self.args.class_per_task*(1+self.args.sess)
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            targets_one_hot = torch.zeros(inputs.shape[0], bi).scatter_(1, targets[:, None], 1)
            inputs, targets_one_hot = inputs.to(self.device), targets_one_hot.to(self.device)

            reptile_grads = {}            
            targets_np = targets.detach().cpu().numpy()
            num_updates = 0
            
            for task_idx in range(1+self.args.sess):
                idx = np.where((targets_np >= task_idx*self.args.class_per_task) & (targets_np < (task_idx+1)*self.args.class_per_task))[0]
                ai, bi = self.args.class_per_task*task_idx, self.args.class_per_task*(task_idx+1)
                
                class_inputs = inputs[idx]
                class_targets_one_hot = targets_one_hot[idx]

                adapted_params = self.model.cloned_state_dict()

                for kr in range(self.args.r):
                    class_outputs = self.model(class_inputs, adapted_params)

                    loss = F.binary_cross_entropy_with_logits(class_outputs[:, ai:bi], class_targets_one_hot[:, ai:bi]) 
                    self.optimizer.zero_grad()

                    grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
                    for (key, val), grad in zip(adapted_params.items(), grads):
                        adapted_params[key] = val - self.args.lr * grad

                for key, val in adapted_params.items():
                    if(num_updates==0):
                        reptile_grads[key] = [val.data]
                    else:
                        reptile_grads[key].append(val.data)

                num_updates += 1
            
            with torch.no_grad():
                for key, p in self.model.named_parameters():
                    alpha = np.exp(-self.args.beta*((1.0*self.args.sess)/self.args.num_task))
                    ll = torch.stack(reptile_grads[key])
                    # alpha = 0.5
                    p.copy_(torch.mean(ll,0)*(alpha) + (1-alpha)* p.data)
                
    def test(self):
        class_acc = {}
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            outputs = self.model(inputs.to(self.device))

            pred = torch.argmax(outputs[:,0:self.args.class_per_task*(1+self.args.sess)], 1, keepdim=False).view(1,-1)
            correct = pred.eq(targets.to(self.device).view(1, -1).expand_as(pred)).view(-1) 
            class_acc = self.get_class_accs(pred, correct, class_acc)
            
        acc_task = self.get_task_accuracies(class_acc)
        print('test_task_accs:', acc_task)
        print('test_class_accs:', class_acc)

