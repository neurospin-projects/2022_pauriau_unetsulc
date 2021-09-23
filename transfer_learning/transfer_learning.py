import json
import numpy as np
import copy
import time
import os
import sigraph
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from soma import aims

#from deepsulci.sulci_labeling.method.unet import UnetSulciLabeling
from deepsulci.deeptools.dataset import extract_data #, SulciDataset
from deepsulci.deeptools.models import UNet3D
from deepsulci.sulci_labeling.analyse.stats import esi_score
from deepsulci.sulci_labeling.method.cutting import cutting
from deepsulci.deeptools.early_stopping import EarlyStopping

from dataset_test import SulciDataset
from fine_tunning import FineTunning

class UnetTransferSulciLabelling(object):

    def __init__(self, graphs, hemi, translation_file, cuda=-1, working_path=None, dict_model={}, dict_trained_model={},
                 dict_names=None, dict_bck2=None, sulci_side_list=None):

        self.graphs = graphs
        self.hemi = hemi
        self.model = None

        #dict_sulci / sslist
        self.dict_bck2 = dict_bck2
        self.dict_names = dict_names
        self.sulci_side_list = sulci_side_list
        if sulci_side_list is not None:
            self.dict_sulci = {sulci_side_list[i]: i for i in range(len(sulci_side_list))}
            if 'background' not in self.dict_sulci:
                self.dict_sulci['background'] = -1
            self.sslist = [ss for ss in sulci_side_list if
                           not ss.startswith('unknown') and not ss.startswith('ventricle')]
        else:
            self.dict_sulci = None
            self.sslist = None
        self.background = -1

        #working path
        if working_path is None :
            self.working_path = os.getcwd()
        else:
            self.working_path = working_path

        #dict model
        if 'name' in dict_model.keys():
            self.model_name = dict_model['name']
        else:
            self.model_name = 'unknown_model'
        if 'training_layers' in dict_model.keys():
            self.training_layers = dict_model['training_layers']
        else:
            self.training_layers = ['final_conv']
        if 'fine_tunning_layers' in dict_model.keys():
            self.fine_tunning_layers = dict_model['fine_tunning_layers']
        else:
            self.fine_tunning_layers = ['decoders.2', 'decoders.1', 'decoders.0']
        if 'num_filter' in dict_model.keys():
            self.num_filter = dict_model['num_filter']
        else:
            self.num_filter = 64
        if 'num_conv' in dict_model.keys():
            self.num_conv = dict_model['num_conv']
        else:
            self.num_conv = 1

        self.dict_trained_model = dict_trained_model

        #results
        self.results = {'lr': [],
                        'momentum': [],
                        'batch_size': [],
                        'epoch_loss_val': [],
                        'epoch_loss_train': [],
                        'epoch_acc_val': [],
                        'epoch_acc_train': [],
                        'best_acc': [],
                        'best_epoch': [],
                        'num_epoch': [],
                        'duration': [],
                        'fine_tunning_epoch': [],
                        'threshold_scores': {},
                        'graphs_train': [],
                        'graphs_test': []
                        }
        self.dict_scores = {}

        # translation file
        if os.path.exists(translation_file):
            self.flt = sigraph.FoldLabelsTranslator()
            self.flt.readLabels(translation_file)
            self.trfile = translation_file
            print('Translation file loaded')
        else:
            self.trfile = None
            print('Translation file not found.')

        # device
        if cuda is -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu", index=cuda)
        print('Working on', self.device)


    def extract_data_from_graphs(self):

        # SULCI SIDE LIST
        print('Creating sulci side list...')
        sulci_side_list = set()
        dict_bck2, dict_names = {}, {}
        for gfile in self.graphs:
            graph = aims.read(gfile)
            if self.trfile is not None:
                self.flt.translate(graph)
            data = extract_data(graph)
            dict_bck2[gfile] = data['bck2']
            dict_names[gfile] = data['names']
            for n in data['names']:
                sulci_side_list.add(n)

        self.sulci_side_list = sorted(list(sulci_side_list))
        print(len(self.sulci_side_list), ' sulci detected')
        self.dict_sulci = {self.sulci_side_list[i]: i for i in range(len(sulci_side_list))}
        if 'background' not in self.dict_sulci:
            self.dict_sulci['background'] = -1
        self.sslist = [ss for ss in sulci_side_list if not ss.startswith('unknown') and not ss.startswith('ventricle')]
        self.dict_bck2 = dict_bck2
        self.dict_names = dict_names


    def load_model(self):
        # MODEL
        # Load file
        print('Network initialization...')
        self.dict_trained_model = self.fill_dict_model(self.dict_trained_model)

        torch.manual_seed(42)

        trained_model = UNet3D(self.dict_trained_model['in_channels'], self.dict_trained_model['out_channels'],
                               final_sigmoid=self.dict_trained_model['final_sigmoid'], interpolate=self.dict_trained_model['interpolate'],
                               conv_layer_order=self.dict_trained_model['conv_layer_order'], init_channel_number=self.dict_trained_model['init_channel_number'])
        trained_model.load_state_dict(torch.load(self.dict_trained_model['model_file'], map_location='cpu'))
        self.model = copy.deepcopy(trained_model)
        if self.num_conv > 1:
            self.model.final_conv = ConvNet(self.dict_trained_model['init_channel_number'], len(self.sulci_side_list), self.num_conv)
        else:
            self.model.final_conv = nn.Conv3d(self.dict_trained_model['init_channel_number'], len(self.sulci_side_list), 1)
        self.model = self.model.to(self.device)

    def fill_dict_model(self, dict_model):
        if 'in_channels' not in dict_model.keys():
            dict_model['in_channels'] = 1
        if 'out_channels' in dict_model.keys():
            if isinstance(dict_model['out_channels'], str):
                param = json.load(open(dict_model['out_channels'], 'r'))
                trained_sulci_side_list = param['sulci_side_list']
                dict_model['out_channels'] = len(trained_sulci_side_list)
        else:
            if self.hemi == 'L':
                path = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_params_left.json'
            else:
                path = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_params_right.json'
            param = json.load(open(path, 'r'))
            trained_sulci_side_list = param['sulci_side_list']
            dict_model['out_channels'] = len(trained_sulci_side_list)
        if 'final_sigmoid' not in dict_model.keys():
            dict_model['final_sigmoid'] = False
        if 'interpolate' not in dict_model.keys():
            dict_model['interpolate'] = True
        if 'conv_layer_order' not in dict_model.keys():
            dict_model['conv_layer_order'] = 'crg'
        if 'init_channel_number' not in dict_model.keys():
            dict_model['init_channel_number'] = 64
        if 'model_file' not in dict_model.keys():
            if self.hemi == 'L':
                dict_model['model_file'] = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_left.mdsm'
            else:
                dict_model['model_file'] = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_right.mdsm'
        return dict_model


    def learning(self, lr, momentum, num_epochs, gfile_list_train, gfile_list_test, batch_size=1, patience={}, save_results=True):

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
            random.seed(42)
            np.random.seed(42)

        # # MODEL # #
        self.load_model()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=0)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        if save_results:
            num_training = len(self.results['lr'])
            self.results['lr'].append(lr)
            self.results['momentum'].append(momentum)
            self.results['batch_size'].append(batch_size)
            self.results['num_epoch'].append(num_epochs)
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
            writer = SummaryWriter(log_dir=log_dir+'/cv_'+str(num_training)) #, comment=)

        # # TRAINING # #
        print('training...')

        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc, epoch_acc = 0., 0.
        best_epoch = 0

        # early stopping
        if 'fine_tunning' in patience.keys():
            fine_tunning = FineTunning(patience=patience['fine_tunning'], save=False)
        if 'early_stopping' in patience.keys():
            es_stop = EarlyStopping(patience=patience['early_stopping'])

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

                    if phase == 'train':
                        for name, parameters in self.model.named_parameters():
                            if np.any([name.startswith(layer) for layer in self.training_layers]):
                                parameters.requires_grad = True
                            else:
                                parameters.requires_grad = False
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    else:
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

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

            print('Epoch took %i s.' % (time.time() - start_time))

            # fine tunning
            if 'fine_tunning' in patience.keys():
                fine_tunning(epoch_loss, self.model)
                if fine_tunning.ft_start:
                    print('\nFine tunning')
                    self.training_layers += self.fine_tunning_layers
                    lr = lr / 10
                    print('Divide learning rate. New value: {}\n'.format(lr))
                    optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
                    if save_results:
                        self.results['fine_tunning_epoch'].append(epoch)
            # early_stopping
            if 'early_stopping' in patience.keys():
                es_stop(epoch_loss, self.model)
                if es_stop.early_stop:
                    print("\nEarly stopping")
                    break

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

    def test_thresholds(self, gfile_list_test, gfile_list_notcut_test, threshold_range, save_results=True):
        print('test thresholds')
        since = time.time()
        for th in threshold_range:
                self.dict_scores[th] = []

        for gfile, gfile_notcut in zip(gfile_list_test, gfile_list_notcut_test):
            # extract data
            graph = aims.read(gfile)
            if self.trfile is not None:
                self.flt.translate(graph)
            data = extract_data(graph)
            nbck = np.asarray(data['nbck'])
            bck2 = np.asarray(data['bck2'])
            names = np.asarray(data['names'])

            graph_notcut = aims.read(gfile_notcut)
            if self.trfile is not None:
                self.flt.translate(graph_notcut)
            data_notcut = extract_data(graph_notcut)
            nbck_notcut = np.asarray(data_notcut['nbck'])
            vert_notcut = np.asarray(data_notcut['vert'])

            # compute labeling
            _, _, yscores = self.labeling(gfile)

            # organize dataframes
            df = pd.DataFrame()
            df['point_x'] = nbck[:, 0]
            df['point_y'] = nbck[:, 1]
            df['point_z'] = nbck[:, 2]
            df.sort_values(by=['point_x', 'point_y', 'point_z'],
                           inplace=True)

            df_notcut = pd.DataFrame()
            nbck_notcut = np.asarray(nbck_notcut)
            df_notcut['vert'] = vert_notcut
            df_notcut['point_x'] = nbck_notcut[:, 0]
            df_notcut['point_y'] = nbck_notcut[:, 1]
            df_notcut['point_z'] = nbck_notcut[:, 2]
            df_notcut.sort_values(by=['point_x', 'point_y', 'point_z'],
                                  inplace=True)
            if (len(df) != len(df_notcut)):
                print()
                print('ERROR no matches between %s and %s' % (
                    gfile, gfile_notcut))
                print('--- Files ignored to fix the threshold')
                print()
            else:
                df['vert_notcut'] = list(df_notcut['vert'])
                df.sort_index(inplace=True)
                for threshold in threshold_range:
                    ypred_cut = cutting(yscores, df['vert_notcut'], bck2, threshold)
                    ypred_cut = [self.sulci_side_list[y] for y in ypred_cut]

                    self.dict_scores[threshold].append((1 - esi_score(
                        names, ypred_cut, self.sslist)) * 100)

        if save_results:
            for th, sc in self.dict_scores.items():
                if th in self.results['threshold_scores'].keys():
                    self.results['threshold_scores'][th].append(sc)
                else:
                    self.results['threshold_scores'][th] = [sc]

        time_elapsed = time.time() - since
        print('Cutting complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def labeling(self, gfile):
        print('Labeling', gfile)

        self.model = self.model.to(self.device)
        self.model.eval()
        bck2 = self.dict_bck2[gfile]
        names = self.dict_names[gfile]
        dataset = SulciDataset(
            [gfile], self.dict_sulci, train=False,
            translation_file=self.trfile,
            dict_bck2={gfile: bck2}, dict_names={gfile: names})
        data = dataset[0]

        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            if bck2 is None:
                bck_T = np.where(np.asarray(labels) != self.background)
            else:
                tr = np.min(bck2, axis=0)
                bck_T = np.transpose(bck2 - tr)
            _, preds = torch.max(outputs.data, 1)
            ypred = preds[0][bck_T[0], bck_T[1], bck_T[2]].tolist()
            ytrue = labels[bck_T[0], bck_T[1], bck_T[2]].tolist()
            yscores = outputs[0][:, bck_T[0], bck_T[1],
                                 bck_T[2]].tolist()
            yscores = np.transpose(yscores)

        return ytrue, ypred, yscores

    def save_data(self, name=None):
        os.makedirs(self.working_path + '/data', exist_ok=True)
        if name is None:
            path_to_save_data = self.working_path + '/data/' + self.model_name + '.json'
        else:
            path_to_save_data = self.working_path + '/data/' + name + '.json'
        data = {'dict_bck2': self.dict_bck2,
                'dict_names': self.dict_names,
                'sulci_side_list': self.sulci_side_list}
        with open(path_to_save_data, 'w') as f:
            json.dump(data, f)
        print('Data saved')

    def save_model(self, name=None):
        os.makedirs(self.working_path + '/models', exist_ok=True)
        if name is None:
            path_to_save_model = self.working_path + '/models/' + self.model_name + '_model.mdsm'
        else:
            os.makedirs(self.working_path + '/models/' + self.model_name + '/', exist_ok=True)
            path_to_save_model = self.working_path + '/models/' + self.model_name + '/' + name + '_model.mdsm'
        torch.save(self.model.state_dict(), path_to_save_model)
        print('Model saved')

    def save_results(self, name=None):
        os.makedirs(self.working_path + '/results', exist_ok=True)
        if name is None:
            path_to_save_results = self.working_path + '/results/' + self.model_name + '.json'
        else:
            path_to_save_results = self.working_path + '/results/' + name + '.json'
        with open(path_to_save_results, 'w') as f:
            json.dump(self.results, f)
        print('Results saved')

    def save_params(self, best_threshold=None, name=None):
        params = {'dict_bck2': self.dict_bck2,
                  'dict_names': self.dict_names,
                  'sulci_side_list': self.sulci_side_list
                 }
        if best_threshold is not None:
            params['cutting_threshold'] = best_threshold
        if os.path.exists(self.working_path + '/models/' + self.model_name + '/'):
            path_to_save_params = self.working_path + '/models/' + self.model_name + '/'
        else:
            path_to_save_params = self.working_path + '/models/'
        if name is None:
            path_to_save_params += self.model_name + '_params.json'
        else:
            path_to_save_params += name + '_params.json'
        with open(path_to_save_params, 'w') as f:
            json.dump(params, f)
        print('Parameters saved')

    def reset_results(self):
        self.results = {'lr': [],
                        'momentum': [],
                        'batch_size': [],
                        'epoch_loss_val': [],
                        'epoch_acc_val': [],
                        'epoch_loss_train': [],
                        'epoch_acc_train': [],
                        'best_acc': [],
                        'best_epoch': [],
                        'num_epoch': [],
                        'fine_tunning_epoch': [],
                        'duration': [],
                        'threshold_scores': {},
                        'graphs_train': [],
                        'graphs_test': [],
                        'val_image_size': [],
                        'train_image_size': []
                        }

    def load_saved_model(self, dict_model):
        dict_model = self.fill_dict_model(dict_model)

        self.model = UNet3D(dict_model['in_channels'], dict_model['out_channels'],
                               final_sigmoid=dict_model['final_sigmoid'],
                               interpolate=dict_model['interpolate'],
                               conv_layer_order=dict_model['conv_layer_order'],
                               init_channel_number=dict_model['init_channel_number'])
        if self.num_conv > 1:
            self.model.final_conv = ConvNet(dict_model['init_channel_number'], dict_model['out_channels'],
                                            self.num_conv)
        else:
            self.model.final_conv = nn.Conv3d(dict_model['init_channel_number'], dict_model['out_channels'],
                                              1)
        self.model.load_state_dict(torch.load(dict_model['model_file'], map_location='cpu'))
        self.model.to(self.device)
        print("Model Loaded !")


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv):
        super(ConvNet, self).__init__()

        fac = (in_channels - out_channels) / (num_conv + 1)
        self.conv_layers = nn.ModuleList([nn.Conv3d(
            in_channels - round(n * fac),
            in_channels - round((n + 1) * fac), 1) for n in range(num_conv + 1)])

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        return x