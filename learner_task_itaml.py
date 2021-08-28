import os
import torch
import torch.optim as optim
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from utils import *


class Learner():
    def __init__(self, model, args, trainloader, testloader, ses):
        self.model=model
        self.args=args
        self.trainloader=trainloader 
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.state= {key:value for key, value in self.args.__dict__.items() if not key.startswith('__') and not callable(key)} 
        self.best_acc = 0 
        self.testloader=testloader

        self.ses = ses
        
        meta_parameters = []
        normal_parameters = []
        for n,p in self.model.named_parameters():
            meta_parameters.append(p)
            p.requires_grad = True
            if("fc" in n):
                normal_parameters.append(p)

        if(self.args.optimizer=="radam"):
            self.optimizer = RAdam(meta_parameters, lr=self.args.lr, betas=(0.9, 0.999), weight_decay=0)
        elif(self.args.optimizer=="adam"):
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
        elif(self.args.optimizer=="sgd"):
            self.optimizer = optim.SGD(meta_parameters, lr=self.args.lr, momentum=0.9, weight_decay=0.001)

    def learn(self):
        for epoch in range(0, self.args.epochs):
            self.adjust_learning_rate(epoch)

            print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (epoch+1, self.args.epochs, self.state['lr'], self.args.sess))

            self.train(self.model, epoch)
            self.test(self.model)

    def train(self, model, epoch):
        model.train()

        bi = self.args.class_per_task*(1+self.args.sess)
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            sessions = []
             
            targets_one_hot = torch.zeros(inputs.shape[0], bi).scatter_(1, targets[:,None], 1)

            inputs, targets_one_hot, targets = inputs.to(self.device), targets_one_hot.to(self.device),targets.to(self.device)

            reptile_grads = {}            
            np_targets = targets.detach().cpu().numpy()
            num_updates = 0
            
            outputs2, _ = model(inputs)
            
            model_base = copy.deepcopy(model)
            for task_idx in range(1+self.args.sess):
                idx = np.where((np_targets>= task_idx*self.args.class_per_task) & (np_targets < (task_idx+1)*self.args.class_per_task))[0]
                ai = self.args.class_per_task*task_idx
                bi = self.args.class_per_task*(task_idx+1)
                
                ii = 0
                if(len(idx)>0):
                    sessions.append([task_idx, ii])
                    ii += 1
                    for i,(p,q) in enumerate(zip(model.parameters(), model_base.parameters())):
                        p=copy.deepcopy(q)
                        
                    class_inputs = inputs[idx]
                    class_targets_one_hot= targets_one_hot[idx]

                    self.args.r = 1
                        
                    for kr in range(self.args.r):
                        _, class_outputs = model(class_inputs)

                        loss = F.binary_cross_entropy_with_logits(class_outputs[:, ai:bi], class_targets_one_hot[:, ai:bi]) 
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    for i,p in enumerate(model.parameters()):
                        if(num_updates==0):
                            reptile_grads[i] = [p.data]
                        else:
                            reptile_grads[i].append(p.data)
                    num_updates += 1
            
            for i,(p,q) in enumerate(zip(model.parameters(), model_base.parameters())):
                alpha = np.exp(-self.args.beta*((1.0*self.args.sess)/self.args.num_task))
                ll = torch.stack(reptile_grads[i])
                p.data = torch.mean(ll,0)*(alpha) + (1-alpha)* q.data  
                
    def test(self, model):
        class_acc = {}
        
        # switch to evaluate mode
        model.eval()
        
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            targets_one_hot = torch.zeros(inputs.shape[0], self.args.num_class).scatter_(1, targets[:,None], 1)
            
            inputs, targets_one_hot, targets = inputs.to(self.device), targets_one_hot.to(self.device), targets.to(self.device)

            outputs2, _ = model(inputs)
            
            pred = torch.argmax(outputs2[:,0:self.args.class_per_task*(1+self.args.sess)], 1, keepdim=False).view(1,-1)
            correct = pred.eq(targets.view(1, -1).expand_as(pred)).view(-1) 

            for i,p in enumerate(pred.view(-1)):
                key = int(p.detach().cpu().numpy())
                if(correct[i]==1):
                    if(key in class_acc.keys()):
                        class_acc[key] += 1
                    else:
                        class_acc[key] = 1
                        
        acc_task = self.get_task_accuracies(class_acc)

        print("\n".join([str(acc_task[k]).format(".4f") for k in acc_task.keys()]) )    
        print(class_acc)

    def meta_test(self, model, memory, inc_dataset):
        model.eval()
        
        base_model = copy.deepcopy(model)
        class_acc = {}
        for task_idx in range(self.args.sess+1):
            
            memory_data, memory_target = np.array(memory[0], dtype="int32"), np.array(memory[1], dtype="int32")
            
            mem_idx = np.where((memory_target>= task_idx*self.args.class_per_task) & (memory_target < (task_idx+1)*self.args.class_per_task))[0]

            meta_memory_data = memory_data[mem_idx]
            
            meta_model = copy.deepcopy(base_model)
            
            meta_loader = inc_dataset.get_custom_loader_idx(meta_memory_data, mode="train", batch_size=64)

            meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)

            meta_model.train()

            ai = self.args.class_per_task*task_idx
            bi = self.args.class_per_task*(task_idx+1)
            print("Training meta tasks:\t" , task_idx)

            #META testing with given knowledge on task
            meta_model.eval()   
            for cl in range(self.args.class_per_task):
                class_idx = cl + self.args.class_per_task*task_idx

                loader = inc_dataset.get_custom_loader_class([class_idx], mode="test", batch_size=10)

                for batch_idx, (inputs, targets) in enumerate(loader):
                    targets_task = targets-self.args.class_per_task*task_idx

                    inputs, targets_task = inputs.to(self.device), targets_task.to(self.device)

                    _, outputs = meta_model(inputs)

                    pred = torch.argmax(outputs[:,ai:bi], 1, keepdim=False).view(1,-1)
                    correct = pred.eq(targets_task.view(1, -1).expand_as(pred)).view(-1) 

                    for i,p in enumerate(pred.view(-1)):
                        key = int(p.detach().cpu().numpy())
                        key = key + self.args.class_per_task*task_idx
                        if(correct[i]==1):
                            if(key in class_acc.keys()):
                                class_acc[key] += 1
                            else:
                                class_acc[key] = 1

            del meta_model
                                
        acc_task = self.get_task_accuracies(class_acc)

        print("\n".join([str(acc_task[k]).format(".4f") for k in acc_task.keys()]) )    
        print(class_acc)

        return acc_task
        
    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']

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