#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
import argparse
import os
import sys
import torch
from pprint import pprint
from torchvision import datasets, transforms
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from training import train_gan
from networks import build_networks, save_networks, get_optimizers
from options import load_options, get_current_epoch
from counterfactual import generate_counterfactual
from comparison import evaluate_with_comparison
import customized_dataloader
from customized_dataloader import msd_net_dataset

def is_true(x):
    return not not x and x.lower().startswith('t')

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Other options can change with every run
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--fold', type=str, default='train', help='Fold [default: train]')
parser.add_argument('--start_epoch', type=int, help='Epoch to start from (defaults to most recent epoch)')
parser.add_argument('--count', type=int, default=1, help='Number of counterfactuals to generate')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
import counterfactual
from networks import build_networks
from options import load_options


# TODO: Right now, to edit cf_speed et al, you need to edit params.json

###################################################################
                            # options #
###################################################################
debug = True
batch_size = 64
nb_classes = 6


#####################################################################
            # Define paths for saving model and data source #
#####################################################################
# Normally, no need to change these
json_data_base_debug = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/" \
                       "data/object_recognition/image_net/derivatives/" \
                       "dataset_v1_3_partition/npy_json_files/2021_02_old"
json_data_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                 "dataset_v1_3_partition/npy_json_files_shuffled/"

if debug:
    train_known_known_path = os.path.join(json_data_base_debug, "debug_known_known.json")
    valid_known_known_path = os.path.join(json_data_base_debug, "debug_known_known.json")
else:
    train_known_known_path = os.path.join(json_data_base, "train_known_known.json")
    valid_known_known_path = os.path.join(json_data_base, "valid_known_known.json")


#######################################################################
# Create dataset and data loader
#######################################################################
# Data transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize])

valid_transform = train_transform

test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(32),
                                     transforms.ToTensor(),
                                     normalize])

#######################################################################
# Create dataset and data loader
#######################################################################
# Training
train_known_known_dataset = msd_net_dataset(json_path=train_known_known_path,
                                            transform=train_transform)
train_known_known_index = torch.randperm(len(train_known_known_dataset))
dataloader = torch.utils.data.DataLoader(train_known_known_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           drop_last=True,
                                           collate_fn=customized_dataloader.collate,
                                           sampler=torch.utils.data.RandomSampler(
                                               train_known_known_index))

# Validation
valid_known_known_dataset = msd_net_dataset(json_path=valid_known_known_path,
                                            transform=valid_transform)
valid_known_known_index = torch.randperm(len(valid_known_known_dataset))
eval_dataloader = torch.utils.data.DataLoader(valid_known_known_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               collate_fn=customized_dataloader.collate,
                                               sampler=torch.utils.data.RandomSampler(
                                                   valid_known_known_index))

start_epoch = options['start_epoch']
options = load_options(options)
options['epoch'] = start_epoch

# Batch size must be large enough to make a square grid visual
options['batch_size'] = nb_classes + 1

networks = build_networks(nb_classes, **options)

for i in range(options['count']):
    counterfactual.generate_open_set(networks, dataloader, **options)
