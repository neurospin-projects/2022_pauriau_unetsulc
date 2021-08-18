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
from deepsulci.deeptools.dataset import extract_data, SulciDataset
from deepsulci.deeptools.models import UNet3D
from deepsulci.sulci_labeling.analyse.stats import esi_score
from deepsulci.sulci_labeling.method.cutting import cutting


class UnetTransferSulciLabelling(object):

    def __init__(self, graphs, hemi, translation_file, cuda=-1, working_path=None, model_name=None,
                 dict_names=None, dict_bck2=None, sulci_side_list=None):

        self.graphs = graphs
        self.hemi = hemi
        self.model = None

        #dict_sulci / sslist
        self.dict_bck2 = dict_bck2
        self.dict_names = dict_names
        self.sulci_side_list = sulci_side_list
        if sulci_side_list is not None :
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
        #model name
        if model_name is None:
            self.model_name = 'unknown_model'
        else:
            self.model_name = model_name

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
        num_channel = 1
        num_filter = 64

        torch.manual_seed(42)

        if self.hemi == 'L':
            model_file = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_left.mdsm'
            with open(
                    '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_params_left.json',
                    'r') as f:
                param = json.load(f)
        else:
            model_file = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_right.mdsm'
            with open(
                    '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_params_right.json',
                    'r') as f:
                param = json.load(f)

        trained_sulci_side_list = param['sulci_side_list']
        trained_model = UNet3D(num_channel, len(trained_sulci_side_list), final_sigmoid=False,
                               init_channel_number=num_filter)
        trained_model.load_state_dict(torch.load(model_file, map_location='cpu'))
        self.model = copy.deepcopy(trained_model)
        self.model.final_conv = nn.Conv3d(num_filter, len(self.sulci_side_list), 1)
        self.model = self.model.to(self.device)


    def learning(self, lr, momentum, num_epochs, gfile_list_train, gfile_list_test, batch_size=1, save_results=True):

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
            img_size = [0, 0, 0]
            for inputs, _ in valdataset:
                size = inputs.size()
                img_size = [np.max([img_size[i], size[i + 1]]) for i in range(len(img_size))]
            print('Image size:', img_size, sep=' ')
            valdataset_resized = SulciDataset(
                gfile_list_test, self.dict_sulci,
                train=False, translation_file=self.trfile,
                dict_bck2=self.dict_bck2, dict_names=self.dict_names, img_size=img_size)
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
            img_size = [0, 0, 0]
            for _ in range(num_epochs):
                for inputs, _ in traindataset:
                    size = inputs.size()
                    img_size = [np.max([img_size[i], size[i + 1]]) for i in range(len(img_size))]
            traindataset_resized = SulciDataset(
                gfile_list_train, self.dict_sulci,
                train=True, translation_file=self.trfile,
                dict_bck2=self.dict_bck2, dict_names=self.dict_names, img_size=img_size)
            trainloader = torch.utils.data.DataLoader(
                traindataset_resized, batch_size=batch_size,
                shuffle=False, num_workers=0)

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

            log_dir = os.path.join(self.working_path + '/tensorboard/' + self.model_name)
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir+'/cv_'+str(num_training)) #, comment=)

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
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    if phase == 'train':
                        for name, parameters in self.model.named_parameters():
                            if name == 'final_conv.weight' or name == 'final_conv.bias':
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
            print('\n')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}, Epoch {}'.format(best_acc, best_epoch))

        if save_results:
            self.results['best_acc'].append(best_acc)
            self.results['best_epoch'].append(best_epoch)
            writer.close()

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def test_thresholds(self, gfile_list_test, gfile_list_notcut_test, threshold_range, save_results=True):
        print('test thresholds')
        since = time.time()
        for th in threshold_range:
            if th not in self.dict_scores.keys():
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
            self.results['threshold_scores'] = self.dict_scores

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
            path_to_save_model = self.working_path + '/models/' + self.model_name + '.mdsm'
        else:
            path_to_save_model = self.working_path + '/models/' + name + '.mdsm'
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
                        'threshold_scores': [],
                        'graphs_train': [],
                        'graphs_test': []
                        }

    def load_saved_model(self, model_file):
        self.load_model()
        self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
        self.model.to(self.device)
        print("Model Loaded !")
