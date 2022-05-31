#!/usr/bin/env python
import argparse
import os
import sys
import torch
from pprint import pprint
from torchvision import datasets, transforms
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from training import train_gan, train_classifier
from networks import build_networks, save_networks, get_optimizers
from options import load_options, get_current_epoch
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

seed = 0
nb_fake = 2
batch_size = 64

if debug:
    nb_classes = 335
else:
    nb_classes = 293

GAN_EPOCHS = 2
CLASSIFIER_EPOCHS = 2
GENERATOR_MODE="open_set"
RESULT_DIR="/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/osrci/debug"


#####################################################################
            # Define paths for saving model and data source #
#####################################################################
# Normally, no need to change these
json_data_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                 "dataset_v1_3_partition/npy_json_files_shuffled/"

train_known_known_path = os.path.join(json_data_base, "train_known_known.json")
valid_known_known_path = os.path.join(json_data_base, "valid_known_known.json")

test_known_known_path_p0 = os.path.join(json_data_base, "test_known_known_part_0.json")
test_known_known_path_p1 = os.path.join(json_data_base, "test_known_known_part_1.json")
test_known_known_path_p2 = os.path.join(json_data_base, "test_known_known_part_2.json")
test_known_known_path_p3 = os.path.join(json_data_base, "test_known_known_part_3.json")

test_unknown_unknown_path = os.path.join(json_data_base, "test_unknown_unknown.json")


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

# Testing
test_known_known_dataset_p0 = msd_net_dataset(json_path=test_known_known_path_p0,
                                               transform=test_transform)
test_known_known_index_p0 = torch.randperm(len(test_known_known_dataset_p0))

test_known_known_dataset_p1 = msd_net_dataset(json_path=test_known_known_path_p1,
                                              transform=test_transform)
test_known_known_index_p1 = torch.randperm(len(test_known_known_dataset_p1))

test_known_known_dataset_p2 = msd_net_dataset(json_path=test_known_known_path_p2,
                                              transform=test_transform)
test_known_known_index_p2 = torch.randperm(len(test_known_known_dataset_p2))

test_known_known_dataset_p3 = msd_net_dataset(json_path=test_known_known_path_p3,
                                              transform=test_transform)
test_known_known_index_p3 = torch.randperm(len(test_known_known_dataset_p3))


test_unknown_unknown_dataset = msd_net_dataset(json_path=test_unknown_unknown_path,
                                               transform=test_transform)
test_unknown_unknown_index = torch.randperm(len(test_unknown_unknown_dataset))

# When doing test, set the batch size to 1 to test the time one by one accurately
test_known_known_loader_p0 = torch.utils.data.DataLoader(test_known_known_dataset_p0,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      sampler=torch.utils.data.RandomSampler(
                                                          test_known_known_index_p0),
                                                      collate_fn=customized_dataloader.collate,
                                                      drop_last=True)

test_known_known_loader_p1 = torch.utils.data.DataLoader(test_known_known_dataset_p1,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         sampler=torch.utils.data.RandomSampler(
                                                             test_known_known_index_p1),
                                                         collate_fn=customized_dataloader.collate,
                                                         drop_last=True)

test_known_known_loader_p2 = torch.utils.data.DataLoader(test_known_known_dataset_p2,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         sampler=torch.utils.data.RandomSampler(
                                                             test_known_known_index_p2),
                                                         collate_fn=customized_dataloader.collate,
                                                         drop_last=True)

test_known_known_loader_p3 = torch.utils.data.DataLoader(test_known_known_dataset_p3,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         sampler=torch.utils.data.RandomSampler(
                                                             test_known_known_index_p3),
                                                         collate_fn=customized_dataloader.collate,
                                                         drop_last=True)

test_unknown_unknown_loader = torch.utils.data.DataLoader(test_unknown_unknown_dataset,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          sampler=torch.utils.data.RandomSampler(
                                                              test_unknown_unknown_index),
                                                          collate_fn=customized_dataloader.collate,
                                                          drop_last=True)


#######################################################################
# Setup Network and training options
#######################################################################
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

networks = build_networks(num_classes=nb_classes, **options)
optimizers = get_optimizers(networks=networks, **options)
start_epoch = get_current_epoch(result_dir=RESULT_DIR) + 1


#######################################################################
# Experiment pipeline
#######################################################################
# 1. Train GAN (E+G+D)
for epoch in range(GAN_EPOCHS):
    train_gan(networks=networks,
              optimizers=optimizers,
              dataloader=dataloader,
              epoch=epoch,
              **options)

    save_networks(networks=networks,
                  epoch=epoch,
                  result_dir=RESULT_DIR)

    # 2. Evaluate on close-set
    # eval_results = evaluate_with_comparison(networks=networks,
    #                                         dataloader=eval_dataloader,
    #                                         **options)
    # print("Eval on close-set: ", eval_results)

# 3. Generate a number of counter-factual images
for i in range(nb_fake):
    counterfactual.generate_open_set(networks=networks,
                                     dataloader=dataloader,
                                     fake_index=i,
                                     save_base=RESULT_DIR,
                                     **options)

# TODO: (???) Automatically label the rightmost column
# (Not sure what the author means. Looks like they just saved
# the fake image again but into a dataset)
# Q: why is the label 0 when saving those fake images??

# 5. Train a new classifier with k+1 classes
for epoch in range(CLASSIFIER_EPOCHS):
    train_classifier(networks=networks,
                     optimizers=optimizers,
                     dataloader=dataloader,
                     fake_img_dir=os.path.join(RESULT_DIR, "images"))

    save_networks(networks, epoch, options['result_dir'])

# TODO: 6. Run data through model - save labels and features
# TODO: Training
train_classifier(networks=networks,
                 optimizers=optimizers,
                 dataloader=dataloader,
                 fake_img_dir=os.path.join(RESULT_DIR, "images"),
                 train_model=False,
                 save_feature=True,
                 feature_name="train",
                 save_feature_path=RESULT_DIR)

# TODO: Validation
train_classifier(networks=networks,
                 optimizers=optimizers,
                 dataloader=eval_dataloader,
                 fake_img_dir=os.path.join(RESULT_DIR, "images"),
                 train_model=False,
                 save_feature=True,
                 feature_name="valid",
                 save_feature_path=RESULT_DIR)

# TODO: Test known
train_classifier(networks=networks,
                 optimizers=optimizers,
                 dataloader=test_known_known_loader_p0,
                 fake_img_dir=os.path.join(RESULT_DIR, "images"),
                 train_model=False,
                 save_feature=True,
                 feature_name="test_p0",
                 save_feature_path=RESULT_DIR)

train_classifier(networks=networks,
                 optimizers=optimizers,
                 dataloader=test_known_known_loader_p1,
                 fake_img_dir=os.path.join(RESULT_DIR, "images"),
                 train_model=False,
                 save_feature=True,
                 feature_name="test_p1",
                 save_feature_path=RESULT_DIR)

train_classifier(networks=networks,
                 optimizers=optimizers,
                 dataloader=test_known_known_loader_p2,
                 fake_img_dir=os.path.join(RESULT_DIR, "images"),
                 train_model=False,
                 save_feature=True,
                 feature_name="test_p2",
                 save_feature_path=RESULT_DIR)

train_classifier(networks=networks,
                 optimizers=optimizers,
                 dataloader=test_known_known_loader_p3,
                 fake_img_dir=os.path.join(RESULT_DIR, "images"),
                 train_model=False,
                 save_feature=True,
                 feature_name="test_p3",
                 save_feature_path=RESULT_DIR)

# TODO: Test unknown
train_classifier(networks=networks,
                 optimizers=optimizers,
                 dataloader=test_unknown_unknown_loader,
                 fake_img_dir=os.path.join(RESULT_DIR, "images"),
                 train_model=False,
                 save_feature=True,
                 feature_name="test_unknown",
                 save_feature_path=RESULT_DIR)



