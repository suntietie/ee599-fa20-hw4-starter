import numpy as np
import os
import os.path as osp
import argparse


parse = argparse.ArgumentParser(description="run on AWS or PC")
Config ={}
# Config['root_path'] = '/Users/tieming/code/dataset/polyvore_outfits'
Config['root_path'] = '/mnt/polyvore_outfits'

Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''
Config['test_category'] ='test_category_hw.txt'
Config['test_category_output'] = 'test_category_predict.txt'


Config['compa_train'] = 'compatibility_train.txt'
Config['compa_valid'] = 'compatibility_valid.txt'
Config['compa_test'] = 'compatibility_test_hw.txt'
Config['train_json'] = 'train.json'
Config['valid_json'] = 'valid.json'
Config['test_json'] = '../test.json'

Config['resnet_pth'] = 'resnet_model.pth'
Config['vgg_pth'] = 'vgg_model.pth'


Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 20
Config['batch_size'] = 64

Config['learning_rate'] = 0.002
Config['num_workers'] = 5

# user-defined model
Config['pretrained'] = False
Config['VGG16'] = False
