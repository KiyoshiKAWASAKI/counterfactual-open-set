#!/usr/bin/env python
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
import counterfactual

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]',
                    default='.')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for [default: 10]')
parser.add_argument('--aux_dataset', help='Path to aux_dataset file [default: None]')

options = vars(parser.parse_args())
options = load_options(options)


###################################################################
                            # options #
###################################################################
debug = True

nb_fake = 2
batch_size = 64
nb_classes = 293

GAN_EPOCHS=2
CLASSIFIER_EPOCHS=2
CF_COUNT=100
GENERATOR_MODE="open_set"
RESULT_DIR="/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/osrci/debug"


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


#######################################################################
# Setup Network and training options
#######################################################################
networks = build_networks(nb_classes, **options)
optimizers = get_optimizers(networks, **options)
start_epoch = get_current_epoch(RESULT_DIR) + 1


#######################################################################
# Experiment pipeline
#######################################################################
# 1. Train GAN (E+G+D)
for epoch in range(GAN_EPOCHS):
    train_gan(networks, optimizers, dataloader, epoch=epoch, **options)
    save_networks(networks, epoch, RESULT_DIR)

    # 2. Evaluate on close-set
    eval_results = evaluate_with_comparison(networks, eval_dataloader, **options)
    pprint(eval_results)

# 3. Generate a number of counter-factual images
for i in range(nb_fake):
    counterfactual.generate_open_set(networks, dataloader, i, RESULT_DIR, **options)

# 4. Automatically label the rightmost column

