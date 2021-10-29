# -*- coding: utf-8 -*-

import json
import numpy as np
import copy
import time
import os
import sigraph
import pandas as pd
import random
#Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
#Brainvisa
from soma import aims
#from deepsulci.sulci_labeling.method.unet import UnetSulciLabeling
from deepsulci.deeptools.dataset import extract_data#, SulciDataset
from deepsulci.deeptools.models import UNet3D
from deepsulci.sulci_labeling.analyse.stats import esi_score
from deepsulci.sulci_labeling.method.cutting import cutting
from deepsulci.deeptools.early_stopping import EarlyStopping

from dataset import SulciDataset
from divide_lr import DivideLr
from pattern_class import UnetPatternSulciLabelling

# -------------------------------------------------------------------------------------------------- #
# Classe permettant d'entraîner le modèle UNET3D sur une cohorte
# --------------------------------------------------------------------------------------------------- #

class UnetTrainingSulciLabelling(UnetPatternSulciLabelling):

    def __init__(self, graphs, hemi, cuda=-1, working_path=None, dict_model={},
                 dict_names=None, dict_bck2=None, sulci_side_list=None):
        
        super().__init__(graphs, hemi, cuda, working_path, dict_model, dict_names, dict_bck2, sulci_side_list)

        #results
        self.results = {'lr': [],
                        'momentum': [],
                        'batch_size': [],
                        'num_epochs': [],
                        'graphs_train': [],
                        'graphs_test': [],
                        'patience': {},
                        'epoch_loss_val': [],
                        'epoch_acc_val': [],
                        'epoch_loss_train': [],
                        'epoch_acc_train': [],
                        'best_acc': [],
                        'best_epoch': [],
                        'divide_lr_epoch': [],
                        'duration': [],
                        'threshold_scores': {}
                        }


    def load_network(self):
        # NETWORK
        # Load file
        print('Network initialization...')

        self.model = UNet3D(self.num_channel,  len(self.sulci_side_list), final_sigmoid=self.final_sigmoid,
                            interpolate=self.interpolate, dropout=0., conv_layer_order=self.conv_layer_order,
                            init_channel_number=self.num_filter)
        if self.num_conv > 1:
            fac = (self.dict_trained_model['init_channel_number'] - len(self.sulci_side_list)) / self.num_conv
            num_channel = self.dict_trained_model['init_channel_number']
            self.model.final_conv = nn.Sequential()
            for n in range(self.num_conv):
                self.model.final_conv.add_module(str(n), nn.Conv3d(num_channel - round(n * fac), num_channel  - round((n + 1) * fac), 1))
        self.model = self.model.to(self.device)


    def learning(self, lr, momentum, num_epochs, gfile_list_train, gfile_list_test, batch_size=1, patience={}, save_results=True):
        #Training
        #Error
        if self.sulci_side_list is None or self.dict_bck2 is None or self.dict_bck2 is None:
            print('Error : extract data from graphs before learning')
            return 1

        # # DATASET / DATALOADERS # #
        print('Extract validation dataloader...')
        valdataset = SulciDataset(
            gfile_list_test, self.dict_sulci,
            train=False, translation_file=self.trfile,
            dict_bck2=self.dict_bck2, dict_names=self.dict_names)

        if batch_size == 1:
            valloader = torch.utils.data.DataLoader(
                valdataset, batch_size=batch_size,
                shuffle=False, num_workers=0)
        else:
            val_img_size = [0, 0, 0]
            for inputs, _ in valdataset:
                size = inputs.size()
                val_img_size = [np.max([val_img_size[i], size[i + 1]]) for i in range(len(val_img_size))]
            print('Val dataset image size:', val_img_size, sep=' ')
            valdataset_resized = SulciDataset(
                gfile_list_test, self.dict_sulci,
                train=False, translation_file=self.trfile,
                dict_bck2=self.dict_bck2, dict_names=self.dict_names, img_size=val_img_size)
            valloader = torch.utils.data.DataLoader(
                valdataset_resized, batch_size=batch_size,
                shuffle=False, num_workers=0)

        print('Extract train dataloader...')
        traindataset = SulciDataset(
            gfile_list_train, self.dict_sulci,
            train=True, translation_file=self.trfile,
            dict_bck2=self.dict_bck2, dict_names=self.dict_names)

        if batch_size == 1:
            trainloader = torch.utils.data.DataLoader(
                traindataset, batch_size=batch_size,
                shuffle=False, num_workers=0)
        else:
            random.seed(42)
            np.random.seed(42)
            train_img_size = [0, 0, 0]
            for _ in range(num_epochs):
                for inputs, _ in traindataset:
                    size = inputs.size()
                    train_img_size = [np.max([train_img_size[i], size[i + 1]]) for i in range(len(train_img_size))]
            print('Train dataset image size:', train_img_size, sep=' ')
            traindataset_resized = SulciDataset(
                gfile_list_train, self.dict_sulci,
                train=True, translation_file=self.trfile,
                dict_bck2=self.dict_bck2, dict_names=self.dict_names, img_size=train_img_size)
            trainloader = torch.utils.data.DataLoader(
                traindataset_resized, batch_size=batch_size,
                shuffle=False, num_workers=0)
            np.random.seed(42)
            random.seed(42)

        # # MODEL # #
        self.load_network()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=0)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        if save_results:
            num_training = len(self.results['lr'])
            self.results['lr'].append(lr)
            self.results['momentum'].append(momentum)
            self.results['batch_size'].append(batch_size)
            self.results['num_epochs'].append(num_epochs)
            self.results['graphs_test'].append(list(gfile_list_test))
            self.results['graphs_train'].append(list(gfile_list_train))
            self.results['patience'] = patience
            if batch_size > 1:
                if num_training == 0:
                    self.results['train_image_size'] = [int(i) for i in train_img_size]
                    self.results['val_image_size'] = [int(i) for i in val_img_size]
                else:
                    self.results['train_image_size'].append([int(i) for i in train_img_size])
                    self.results['val_image_size'].append([int(i) for i in val_img_size])

            log_dir = os.path.join(self.working_path + '/tensorboard/' + self.model_name)
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir+'/cv'+str(num_training)) #, comment=)

        # early stopping
        if "early_stopping" in patience.keys():
            es_stop = EarlyStopping(patience=patience['early_stopping'])
        if "divide_lr" in patience.keys():
            divide_lr = DivideLr(patience=patience['divide_lr'])


        # # TRAINING # #
        print('training...')

        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc, epoch_acc = 0., 0.
        best_epoch = 0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            start_time = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0

                # compute dataloader
                dataloader = trainloader if phase == 'train' else valloader

                # Iterate over data.
                y_pred, y_true = [], []
                for batch, (inputs, labels) in enumerate(dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    y_pred.extend(preds[labels != self.background].tolist())
                    y_true.extend(labels[labels != self.background].tolist())

                    if batch_size > 1:
                        print('Batch n°{:.0f}/{:.0f} || Loss: {:.4f}'.format(batch+1, np.ceil(len(dataloader.dataset)/batch_size), loss.item()))

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = 1 - esi_score(
                    y_true, y_pred,
                    [self.dict_sulci[ss] for ss in self.sslist])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if save_results:
                    writer.add_scalar('Loss/'+phase, epoch_loss, epoch)
                    writer.add_scalar('Accuracy/'+phase, epoch_acc, epoch)
                    if epoch == 0:
                        self.results['epoch_loss_'+phase].append([epoch_loss])
                        self.results['epoch_acc_'+phase].append([epoch_acc])
                    else:
                        self.results['epoch_loss_'+phase][num_training].append(epoch_loss)
                        self.results['epoch_acc_'+phase][num_training].append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            # divide_lr
            if 'divide_lr' in patience.keys():
                divide_lr(epoch_loss, self.model)
                if divide_lr.divide_lr:
                    lr = lr / 10
                    print('\tDivide learning rate. New value: {}'.format(lr))
                    optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
                    self.results['divide_lr_epoch'].append(epoch)
            # early_stopping
            if "early_stopping" in patience.keys():
                es_stop(epoch_loss, self.model)
                if es_stop.early_stop:
                    print("Early stopping")
                    break

            print('Epoch took %i s.' % (time.time() - start_time))
            print('\n')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}, Epoch {}'.format(best_acc, best_epoch))

        if save_results:
            self.results['best_acc'].append(best_acc)
            self.results['best_epoch'].append(best_epoch)
            self.results['duration'].append(time_elapsed)
            writer.close()

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def reset_results(self):
        self.results = {'lr': [],
                        'momentum': [],
                        'batch_size': [],
                        'num_epochs': [],
                        'graphs_train': [],
                        'graphs_test': [],
                        'patience': {},
                        'train_image_size': [],
                        'val_image_size': [],
                        'epoch_loss_val': [],
                        'epoch_acc_val': [],
                        'epoch_loss_train': [],
                        'epoch_acc_train': [],
                        'best_acc': [],
                        'best_epoch': [],
                        'divide_lr_epoch': [],
                        'duration': [],
                        'threshold_scores': {}
                        }


