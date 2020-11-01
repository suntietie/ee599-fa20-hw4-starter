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


Config['use_cuda'] = True
Config['debug'] = True
Config['num_epochs'] = 20
Config['batch_size'] = 64

Config['learning_rate'] = 0.001
Config['num_workers'] = 5

# user-defined model
Config['pretrained'] = False
Config['VGG16'] = True
