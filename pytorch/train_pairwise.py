import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp

from utils import Config
from model import model_pretrained, MyVgg11, net16, net16_pair
from data import get_pairloader
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot(x_list, y_list, fname, num_epochs=Config['num_epochs']):
    l = [i for i in range(1, len(x_list)+1)]
    new_ticks=np.linspace(0,num_epochs,5)
    plt.plot(l, x_list,label="Training set")
    plt.plot(l, y_list,label="Test set")

    plt.xticks(new_ticks)
    plt.title("Accuracy Performance Versus Epoch")
    plt.legend(labels=["Training set", "Test set"],loc='best')
    plt.xlabel("Epoches")
    plt.ylabel("Accuracy")
    plt.savefig(fname=fname)
    plt.close()
    return 


def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):

    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):

                labels_temp = np.array(labels)
                labels = torch.Tensor(labels_temp.astype('long'))
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view(-1, 1)
                optimizer.zero_grad()

                # only work on the lr_scheduler
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                pred = outputs
                pred[pred >= 0.5] = 1.0
                pred[pred <= 0.5] = 0.0
                running_corrects += torch.sum(pred==labels.data)


            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                acc_train_list.append(epoch_acc)
            if phase == 'test':
                acc_test_list.append(epoch_acc)
            
            if phase=='test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # save model.pth - dictionary (best_model_wts)
        torch.save(best_model_wts, osp.join(Config['root_path'], Config['checkpoint_path'], 'pairwise_model.pth'))
        print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'pairwise_model.pth')))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))



if __name__=='__main__':

    '''Hyper parameters of the dataset:

    dataiter = iter(dataloaders["train"])
    images, labels = dataiter.next()
    >>> images.shape
    torch.Size([64, 3, 224, 224])
    >>> labels.shape
    torch.Size([64])
    >>> classes
    153
    '''

    
    dataloaders, classes, dataset_size = get_pairloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    # acc_list - global variables
    acc_train_list = []
    acc_test_list = []


    model = net16_pair
        
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config['learning_rate'], weight_decay=0.0001)
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')        

    train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)
    plot(acc_train_list, acc_test_list, "pairwise.jpg", num_epochs=Config['num_epochs'])  
