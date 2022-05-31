import time
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable

from vector import make_noise
from dataloader import FlexibleCustomDataloader
import imutil
from logutil import TimeSeries

from gradient_penalty import calc_gradient_penalty


log = TimeSeries('Training GAN')


def train_gan(networks, optimizers, dataloader, epoch=None, **options):
    for net in networks.values():
        net.train()
    netE = networks['encoder']
    netD = networks['discriminator']
    netG = networks['generator']
    netC = networks['classifier_k']
    optimizerE = optimizers['encoder']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerC = optimizers['classifier_k']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    latent_size = options['latent_size']

    # for i in tqdm(range(len(dataloader))):
    for i in tqdm(range(5)):
        batch = next(iter(dataloader))

        images = batch["imgs"]
        images = images.cuda()
        images = Variable(images)
        images = images[:, :, :32, :]

        class_labels = batch["labels"]
        class_labels = class_labels.cuda(async=True)
        labels = Variable(class_labels)

        #ac_scale = random.choice([1, 2, 4, 8])
        ac_scale = 4
        sample_scale = 4
        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Classify sampled images as fake
        noise = make_noise(batch_size, latent_size, sample_scale)
        fake_images = netG(noise, sample_scale)
        logits = netD(fake_images)[:,0]
        # print(logits.shape)
        loss_fake_sampled = F.softplus(logits).mean()
        log.collect('Discriminator Sampled', loss_fake_sampled)
        loss_fake_sampled.backward()

        """
        # Classify autoencoded images as fake
        more_images, more_labels = dataloader.get_batch()
        more_images = Variable(more_images)
        fake_images = netG(netE(more_images, ac_scale), ac_scale)
        logits_fake = netD(fake_images)[:,0]
        #loss_fake_ac = F.softplus(logits_fake).mean() * options['discriminator_weight']
        loss_fake_ac = logits_fake.mean() * options['discriminator_weight']
        log.collect('Discriminator Autoencoded', loss_fake_ac)
        loss_fake_ac.backward()
        """

        # Classify real examples as real
        logits = netD(images)[:,0]
        loss_real = F.softplus(-logits).mean() * options['discriminator_weight']
        #loss_real = -logits.mean() * options['discriminator_weight']
        loss_real.backward()
        log.collect('Discriminator Real', loss_real)

        # print("Real image: ", images.shape)
        # print("Fake image: ", fake_images.shape)

        gp = calc_gradient_penalty(netD, images.data, fake_images.data)
        gp.backward()
        log.collect('Gradient Penalty', gp)

        optimizerD.step()

        ############################

        ############################
        # Generator Update
        ###########################
        netG.zero_grad()

        """
        # Minimize fakeness of sampled images
        noise = make_noise(batch_size, latent_size, sample_scale)
        fake_images_sampled = netG(noise, sample_scale)
        logits = netD(fake_images_sampled)[:,0]
        errSampled = F.softplus(-logits).mean() * options['generator_weight']
        errSampled.backward()
        log.collect('Generator Sampled', errSampled)
        """

        # Minimize fakeness of autoencoded images
        fake_images = netG(netE(images, ac_scale), ac_scale)
        logits = netD(fake_images)[:,0]
        #errG = F.softplus(-logits).mean() * options['generator_weight']
        errG = -logits.mean() * options['generator_weight']
        errG.backward()
        log.collect('Generator Autoencoded', errG)

        optimizerG.step()

        ############################
        # Autoencoder Update
        ###########################
        netG.zero_grad()
        netE.zero_grad()

        # Minimize reconstruction loss
        reconstructed = netG(netE(images, ac_scale), ac_scale)
        err_reconstruction = torch.mean(torch.abs(images - reconstructed)) * options['reconstruction_weight']
        err_reconstruction.backward()
        log.collect('Pixel Reconstruction Loss', err_reconstruction)

        optimizerE.step()
        optimizerG.step()
        ###########################

        ############################
        # Classifier Update
        ############################
        netC.zero_grad()

        # Classify real examples into the correct K classes with hinge loss
        classifier_logits = netC(images)
        classifier_logits = torch.reshape(classifier_logits, (classifier_logits.shape[1], classifier_logits.shape[0]))
        errC = F.softplus(classifier_logits*labels.type(torch.cuda.FloatTensor)).mean()
        errC.backward()
        log.collect('Classifier Loss', errC)

        optimizerC.step()
        ############################

        # Keep track of accuracy on positive-labeled examples for monitoring
        # log.collect_prediction('Classifier Accuracy', netC(images), labels)
        #log.collect_prediction('Discriminator Accuracy, Real Data', netD(images), labels)

        # log.print_every()

        if i % 100 == 1:
            fixed_noise = make_noise(batch_size, latent_size, sample_scale, fixed_seed=42)
            demo(networks, images, fixed_noise, ac_scale, sample_scale, result_dir, epoch, i)
    return True




def demo(networks, images, fixed_noise, ac_scale, sample_scale, result_dir, epoch=0, idx=0):
    netE = networks['encoder']
    netG = networks['generator']

    def image_filename(*args):
        image_path = os.path.join(result_dir, 'images')
        name = '_'.join(str(s) for s in args)
        name += '_{}'.format(int(time.time() * 1000))
        return os.path.join(image_path, name) + '.jpg'

    demo_fakes = netG(fixed_noise, sample_scale)
    img = demo_fakes.data[:16]

    filename = image_filename('samples', 'scale', sample_scale)
    caption = "S scale={} epoch={} iter={}".format(sample_scale, epoch, idx)
    imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)

    aac_before = images[:8]
    aac_after = netG(netE(aac_before, ac_scale), ac_scale)
    img = torch.cat((aac_before, aac_after))

    filename = image_filename('reconstruction', 'scale', ac_scale)
    caption = "R scale={} epoch={} iter={}".format(ac_scale, epoch, idx)
    imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)




def train_classifier(networks,
                     optimizers,
                     dataloader,
                     fake_img_dir,
                     train_model=True,
                     save_feature=False,
                     feature_name=None,
                     save_feature_path=None):

    if train_model:
        for net in networks.values():
            net.train()
    else:
        for net in networks.values():
            net.eval()

    netC = networks['classifier_kplusone']
    optimizerC = optimizers['classifier_kplusone']

    # TODO: Get fake image batches
    aux_batch_name_list = os.listdir(fake_img_dir)
    aux_batch_name_list = [f for f in aux_batch_name_list if f.endswith(".npy")]
    nb_aux_batch = len(aux_batch_name_list)

    print("Total number of real img batches: ", len(dataloader))
    print("Total number of fake img batches: ", nb_aux_batch)

    # TODO: Generate a list for fake batches
    aux_index_list = np.random.randint(low=0,
                                       high=nb_aux_batch,
                                       size=len(dataloader))

    features = []
    aux_feats = []
    save_labels = []

    # for i in tqdm(range(len(dataloader))):
    for i in tqdm(range(5)):
        batch = next(iter(dataloader))

        images = batch["imgs"]
        images = images.cuda()
        images = Variable(images)
        images = images[:, :, :32, :]

        class_labels = batch["labels"]
        class_labels = class_labels.cuda(async=True)
        labels = Variable(class_labels)

        ############################
        # Classifier Update
        ############################
        netC.zero_grad()

        # Classify real examples into the correct K classes
        classifier_logits = netC(images)
        augmented_logits = F.pad(classifier_logits, (0,1))
        # _, labels_idx = labels.max(dim=1)
        # labels_idx = labels
        if train_model:
            errC = F.nll_loss(F.log_softmax(augmented_logits, dim=1), labels.type(torch.cuda.LongTensor))
            errC.backward()

        # print("images: ", images.shape)
        # print("classifier_logits: ", classifier_logits.shape)
        # print("augmented_logits: ", augmented_logits.shape)

        # TODO: Load a batch of fake images
        aux_images = np.load(os.path.join(fake_img_dir,
                                          aux_batch_name_list[aux_index_list[i]]))
        aux_images = aux_images.transpose((0, 3, 1, 2))
        aux_images = Variable(torch.from_numpy(aux_images))

        # Classify aux_dataset examples as open set
        classifier_logits = netC(aux_images.cuda())
        augmented_logits = F.pad(classifier_logits, (0,1))

        # print("aux image: ", aux_images.shape)
        # print("classifier_logits: ", classifier_logits.shape)
        # print("augmented_logits: ", augmented_logits.shape)

        log_soft_open = F.log_softmax(augmented_logits, dim=1)[:, -1]
        errOpenSet = -log_soft_open.mean()

        if save_feature:
            logits = classifier_logits.tolist()
            aug_logist = augmented_logits.tolist()
            labels = labels.tolist()

            for one_logit in logits:
                features.append(one_logit)

            for one_logit in aug_logist:
                aux_feats.append(one_logit)

            for one_label in labels:
                save_labels.append(one_label)

        if train_model:
            errOpenSet.backward()
            optimizerC.step()

    if save_feature:
        features_np = np.asarray(features)
        labels_np = np.asarray(save_labels)
        aug_features_np = np.asarray(aux_feats)

        print(features_np.shape)
        print(labels_np.shape)
        print(aug_features_np.shape)

        np.save(save_feature_path + "/" + feature_name + "_features.npy", features_np)
        np.save(save_feature_path + "/" + feature_name + "_aug_features.npy", aug_features_np)
        np.save(save_feature_path + "/" + feature_name + "_labels.npy", labels_np)

    return True
