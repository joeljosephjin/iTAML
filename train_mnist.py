import random
import torch
import torch.utils.data as data
import numpy as np

import torch
import sys

from utils import BasicNet1
from learner_task_itaml import Learner
import incremental_dataloader as data

class args:
    data_path = "../Datasets/MNIST/"
    num_class = 10
    class_per_task = 2
    num_task = 5
    test_samples_per_class = 1000
    dataset = "mnist"
    optimizer = 'sgd'
    
    epochs = 20
    lr = 1e-4
    train_batch = 256
    test_batch = 256
    workers = 2
    sess = 0
    schedule = [5,10,15]
    gamma = 0.5
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 0.5
    r = 1
    
# Use CUDA
use_cuda = torch.cuda.is_available()
seed = random.randint(1, 10000)
seed = 2481 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

model = BasicNet1(args, 0).cuda() 

inc_dataset = data.IncrementalDataset(
                    dataset_name=args.dataset,
                    args = args,
                    random_order=args.random_classes,
                    shuffle=True,
                    seed=1,
                    batch_size=args.train_batch,
                    workers=args.workers,
                    validation_split=args.validation,
                    increment=args.class_per_task,
                )
    
start_sess = int(sys.argv[1])
memory = None

for ses in range(start_sess, args.num_task):
    args.sess=ses 
    
    if(ses==0):
        # args.epochs = 5
        args.epochs = 5
    if(ses==4):
        args.lr = 0.05
    if(start_sess==ses and start_sess!=0): 
        inc_dataset._current_task = ses
        inc_dataset.sample_per_task_testing = sample_per_task_testing
        args.sample_per_task_testing = sample_per_task_testing
    
    if ses>0: 
        # args.epochs = 20
        args.epochs = 5
        
    task_info, train_loader, val_loader, test_loader, for_memory = inc_dataset.new_task(memory)

    print('task_info:', task_info)
    print('sample_per_task_testing:', inc_dataset.sample_per_task_testing)

    args.sample_per_task_testing = inc_dataset.sample_per_task_testing
    
    main_learner=Learner(model=model,args=args,trainloader=train_loader, testloader=test_loader, use_cuda=use_cuda, ses=ses)
    
    main_learner.learn()

    memory = inc_dataset.get_memory(memory, for_memory)

    acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)

