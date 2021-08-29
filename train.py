import random
import torch
import torch.utils.data as data
import numpy as np

import torch
import sys

from utils import Net
from learner import Learner
import incremental_dataloader as data

class args:
    data_path = "../Datasets/MNIST/"
    num_class = 10
    class_per_task = 2
    num_task = 5
    test_samples_per_class = 1000
    dataset = "mnist"
    
    # epochs = 5
    epochs = 20
    lr = 0.05
    # lr = 0.005
    train_batch = 256
    test_batch = 256
    workers = 16
    sess = 0
    schedule = [5,10,15]
    # gamma = 0.5
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 0.5
    # beta = -5
    r = 5

    start_sess = 0
    
seed = 2481
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = Net().to(device)

inc_dataset = data.IncrementalDataset(dataset_name=args.dataset, args=args, random_order=args.random_classes,
                        shuffle=True, seed=1, batch_size=args.train_batch, workers=args.workers,
                        validation_split=args.validation, increment=args.class_per_task)
    
memory = None
for ses in range(args.start_sess, args.num_task):
    args.sess=ses

    if ses == 0:
        args.epochs=5
    else:
        args.epochs=10
    
    task_info, train_loader, val_loader, test_loader, for_memory = inc_dataset.new_task(memory)
    memory = inc_dataset.get_memory(memory, for_memory)

    args.sample_per_task_testing = inc_dataset.sample_per_task_testing
    
    main_learner=Learner(model=model, args=args, trainloader=train_loader, testloader=test_loader)
    
    for epoch in range(0, args.epochs):
        main_learner.adjust_learning_rate(epoch)

        print('\nEpoch: [%d | %d] Sess: %d' % (epoch+1, args.epochs, args.sess))

        main_learner.train()
        main_learner.test()

