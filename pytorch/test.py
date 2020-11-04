import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder
from model import model_pretrained, net16
from data import polyvore_test_write
import os
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image

from utils import Config



resnet = True


dataset = polyvore_dataset()
transforms = dataset.get_data_transforms()
test_output = open(osp.join(Config['root_path'], Config['test_category_output']), 'w')
device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')


if resnet == True:
    model = model_pretrained
    num_ftrs = model_pretrained.fc.in_features
    model.fc = nn.Linear(num_ftrs, 153)

    model.load_state_dict(torch.load(osp.join(Config['root_path'], Config['resnet_pth'])))
    model.to(device)
else:
    model = net16
    model.load_state_dict(torch.load(osp.join(Config['root_path'], Config['vgg_pth'])))



model.eval()
X_test, y_test = dataset.create_testset()
test_dataset = polyvore_test_write(X_test, y_test, transforms['test'])
dataloader = DataLoader(test_dataset, shuffle=False)
for input_name, inputs, labels in tqdm(dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    _, pred = torch.max(outputs, 1)

    line = ""
    line = str(input_name) +' '+ str(pred) +' '+ str(labels) + '\n'
    test_output.writeline(line)

test_output.close()


    