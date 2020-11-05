import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

from utils import Config


class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.X_train, self.X_test, self.y_train, self.y_test, self.classes = self.create_dataset()
        self.test_dir = Config['test_category']


    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms



    def create_dataset(self):
        # map id to category
        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for k, v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']

        # create X, y pairs
        files = os.listdir(self.image_dir)
        X = []; y = []
        for x in files:
            if x[:-4] in id_to_category:
                X.append(x)
                y.append(int(id_to_category[x[:-4]]))

        y = LabelEncoder().fit_transform(y)
        print('len of X: {}, # of categories: {}'.format(len(X), max(y) + 1))

        # split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test, max(y) + 1


    def create_testset(self):

        meta_file = open(osp.join(self.root_dir, Config['meta_file']), 'r')
        meta_json = json.load(meta_file)
        id_to_category = {}
        for k, v in meta_json.items():
            id_to_category[k] = v['category_id'] 
         
        # add test file 
        test_file = open(osp.join(self.root_dir, self.test_dir), 'r')
        test_list = []
        for line in test_file.readlines():
            line = line.strip()
            test_list.append(line)
        # create X, y pairs
        files_list = os.listdir(self.image_dir)
        files_set = set(map(lambda x: x[:-4], files_list))
        X = []; y = []
        for x in test_list:
            if x in files_set and x in id_to_category:
                X.append(x+".jpg")
                y.append(int(id_to_category[x]))
        y = LabelEncoder().fit_transform(y)
        print('len of test set X: {}, # of categories: {}'.format(len(X), max(y) + 1))
        
        return X, y


# For category classification
class polyvore_train(Dataset):

    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_train[item])
        return self.transform(Image.open(file_path)),self.y_train[item]




class polyvore_test(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')


    def __len__(self):
        return len(self.X_test)


    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_test[item])
        return self.transform(Image.open(file_path)), self.y_test[item]


class polyvore_test_write(Dataset):
    def __init__(self, X_test, y_test, transform):
        self.X_test = X_test
        self.y_test = y_test
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')


    def __len__(self):
        return len(self.X_test)


    def __getitem__(self, item):
        file_path = osp.join(self.image_dir, self.X_test[item])
        return self.X_test[item], self.transform(Image.open(file_path)), self.y_test[item]


def get_dataloader(debug, batch_size, num_workers):
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes = dataset.create_dataset()

    if debug==True:
        train_set = polyvore_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_test(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    else:
        train_set = polyvore_train(X_train, y_train, transforms['train'])
        test_set = polyvore_test(X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    datasets = {'train': train_set, 'test': test_set}
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, classes, dataset_size




########################################################################
# For Pairwise Compatibility Classification

class polyvore_pairset:

    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = osp.join(self.root_dir, 'images')
        self.transforms = self.get_data_transforms()
        # self.pair_train = Config['pair_train']
        # self.pair_valid = Config['pair_valid']

    def get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        return data_transforms

    def create_dataset(self, pair_set):
        pair_list = []; pair_label = []
        pair_open =  open(osp.join(self.root_dir, pair_set), 'r')
        for p in pair_open:
            p = p.strip('\n')
            temp_list = p.split(" ")
            pair_list.append([temp_list[1], temp_list[2]])
            pair_label.append(temp_list[0])
        # print(pair_list)
        # create X, y pairs
        files_list = os.listdir(self.image_dir)
        files_set = set(map(lambda x: x[:-4], files_list))
        X = []; y = []
        for i in range(len(pair_list)):
            if pair_list[i][0] in files_set and pair_list[i][1] in files_set:
                    X.append([pair_list[i][0]+".jpg", pair_list[i][1]+".jpg"])
                    y.append(pair_label[i])
        print('len of train set X: {}'.format(len(X)))

        return X, y

    def create_testset(self):
        # pairs without label
        pair_list = []
        with open(osp.join(self.root_dir, Config['pair_test']), 'r') as pair_open:
            line = pair_open.readline()
            temp_list = line.strip().split(" ")
            pair_list.append([temp_list[1], temp_list[2]])

        # create X pairs
        files_list = os.listdir(self.image_dir)
        files_set = set(map(lambda x: x[:-4], files_list))
        X = []
        for i in range(len(pair_list)):
            if pair_list[i][0] in files_set and pair_list[i][1] in files_set:
                    X.append([pair_list[i][0]+".jpg", pair_list[i][1]+".jpg"])

        print('len of test set X: {}'.format(len(X)))

        return X


# For pairset determination
class polyvore_pair_train(Dataset):

    def __init__(self, X_train, y_train, transform):
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform
        self.image_dir = osp.join(Config['root_path'], 'images')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        file_path1 = osp.join(self.image_dir, self.X_train[item][0])
        file_path2 = osp.join(self.image_dir, self.X_train[item][0])
        new_X = torch.cat((self.transform(Image.open(file_path1)), self.transform(Image.open(file_path2))), 0)
        return new_X, self.y_train[item]


def get_pairloader(debug, batch_size, num_workers):
    dataset = polyvore_pairset()
    transforms = dataset.get_data_transforms()
    X_train, y_train = dataset.create_dataset(Config['pair_train'])
    X_valid, y_valid = dataset.create_dataset(Config['pair_valid'])
    
    if debug==True:
        train_set = polyvore_pair_train(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set = polyvore_pair_train(X_valid[:100], y_valid[:100], transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_valid)}
        datasets = {'train': train_set, 'test': test_set}

    else:
        train_set = polyvore_pair_train(X_train, y_train, transform=transforms['train'])
        test_set = polyvore_pair_train(X_valid, y_valid, transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_valid)}
        train_indices = torch.randperm(len(train_set))[:200000]
        
        datasets = {'train': train_set, 'test': test_set}
        datasets['train'] = Subset(train_set, train_indices)
    
    
    dataloaders = {x: DataLoader(datasets[x],
                                 shuffle=True if x=='train' else False,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
                                 for x in ['train', 'test']}
    return dataloaders, 1, dataset_size    
