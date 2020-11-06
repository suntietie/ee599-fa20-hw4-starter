import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from model import mobile
from data import polyvore_test_write, polyvore_test_pair, polyvore_pairset
import os
import os.path as osp
import json
from tqdm import tqdm

from utils import Config
import copy


mobilenet = True


dataset = polyvore_pairset()
transforms = dataset.get_data_transforms()
device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

with open(osp.join(Config['root_path'], Config['test_pairwise_output']), 'w') as test_output:

    # model = net16_pair
    model = mobile
    model.features[0][0] = nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    fc_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(fc_features, 1),
        nn.Sigmoid())

    model.load_state_dict(torch.load(osp.join(Config['root_path'], Config['pairwise_pth'])))
    model.to(device)


    model.eval()
    X_test = dataset.create_testset()
    test_dataset = polyvore_test_pair(X_test, transforms['test'])
    dataloader = DataLoader(test_dataset, shuffle=False, num_workers=Config['num_workers'])
    for input_names, inputs in tqdm(dataloader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        pred = outputs
        out = outputs[0][0].item()
        pred[pred >= 0.5] = 1.0
        pred[pred <= 0.5] = 0.0
        line = ""
        line = str(input_names[0][0][:-4]) +' '+ str(input_names[1][0][:-4]) +' ' + str(out) +' '+ str(pred[0][0].item()) + '\n'
        test_output.write(line)


    