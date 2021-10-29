# -*- coding: utf-8 -*-

import json
import numpy as np
import copy
import time
import os
import os.path as op
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
from deepsulci.deeptools.dataset import extract_data
from deepsulci.deeptools.models import UNet3D
from deepsulci.sulci_labeling.analyse.stats import esi_score
from deepsulci.sulci_labeling.method.cutting import cutting
from deepsulci.deeptools.early_stopping import EarlyStopping

from dataset import SulciDataset

# -------------------------------------------------------------------------------------------------- #
# Classe Parent des classes UnetTrainingSulciLabelling et UnetTransfertSulciLabelling
# Classe alternative à la classe UnetSulciLabelling de Brainvisa (deepsulci/sulci_labelling/method/unet)
# --------------------------------------------------------------------------------------------------- #

class UnetPatternSulciLabelling(object):

    def __init__(self, graphs, hemi, cuda=-1, working_path=None, dict_model={},
                 dict_names=None, dict_bck2=None, sulci_side_list=None):

        #graphs
        self.graphs = graphs
        self.hemi = hemi

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
        if working_path is None:
            self.working_path = os.getcwd()
        else:
            self.working_path = working_path

        #model
        self.model = None
        #dict_model
        self.dict_model = dict_model
        if 'name' in dict_model.keys():
            self.model_name = dict_model['name']
            print('Model name: ', self.model_name)
        else:
            self.model_name = 'UnknownModel_hemi' + hemi
        if 'num_filter' in dict_model.keys():
            self.num_filter = dict_model['num_filter']
            print('Number of filters : ', self.num_filter)
        else:
            self.num_filter = 64
        if 'num_channel' in dict_model.keys():
            self.num_channel = dict_model['num_channel']
            print('Number of channels : ', self.num_channel)
        else:
            self.num_channel = 1
        if 'interpolate' in dict_model.keys():
            self.interpolate = dict_model['interpolate']
            print('Interpolate : ', self.interpolate)
        else:
            self.interpolate = True
        if 'final_sigmoid' in dict_model.keys():
            self.final_sigmoid = dict_model['final_sigmoid']
            print('Final Sigmoid : ', self.final_sigmoid)
        else:
            self.final_sigmoid = False
        if 'conv_layer_order' in dict_model.keys():
            self.conv_layer_order = dict_model['conv_layer_order']
            print('Convolutional Layer Order : ', self.conv_layer_order)
        else:
            self.conv_layer_order = 'crg'
        if 'num_conv' in dict_model.keys():
            self.num_conv = dict_model['num_conv']
        else:
            self.num_conv = 1

        #results
        self.results = {}
        self.dict_scores = {}

        # translation file
        self.trfile = None

        # device
        if cuda is -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu", index=cuda)
        print('Working on', self.device)

    def extract_data_from_graphs(self):
        #Création de la sucli_side_list, dict_names et dict_bck2 à partir des graphes
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

    def fill_dict_model(self, dict_model):
        # Auto-complétion du dict_model avec les paramètres par défaut
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
                dict_model[
                    'model_file'] = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_left.mdsm'
            else:
                dict_model[
                    'model_file'] = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_right.mdsm'
        if 'num_conv' not in dict_model.keys():
            dict_model['num_conv'] = 1
        return dict_model

    def test_thresholds(self, gfile_list_test, gfile_list_notcut_test, threshold_range, save_results=True):
        # Application des cutting threshold et sauvegarde des scores
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

    def labeling(self, gfile, bck2=None, names=None, imgsize=None):
        # Labellisation automatique du graphe gfile avec le modèle
        print('Labeling', gfile)
        self.model = self.model.to(self.device)
        self.model.eval()
        if bck2 is None:
            bck2 = self.dict_bck2[gfile]
        if names is None:
            names = self.dict_names[gfile]
        dataset = SulciDataset(
            [gfile], self.dict_sulci, train=False,
            translation_file=self.trfile,
            dict_bck2={gfile: bck2}, dict_names={gfile: names}, img_size=imgsize)
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
        # Sauvegarde de la sucli_side_list, dict_names et dict_bck2
        os.makedirs(op.join(self.working_path, 'data'), exist_ok=True)
        if name is None:
            path_to_save_data = op.join(self.working_path, 'data', self.model_name + '.json')
        else:
            path_to_save_data = op.join(self.working_path, 'data', name + '_data.json')
        data = {'dict_bck2': self.dict_bck2,
                'dict_names': self.dict_names,
                'sulci_side_list': self.sulci_side_list}
        with open(path_to_save_data, 'w') as f:
            json.dump(data, f)
        print('Data saved')

    def save_model(self, name=None):
        # Sauvegarde du modèle
        os.makedirs(op.join(self.working_path, 'models'), exist_ok=True)
        if name is None:
            path_to_save_model = op.join(self.working_path, 'models', self.model_name + '_model.mdsm')
        else:
            os.makedirs(op.join(self.working_path, 'models', self.model_name), exist_ok=True)
            path_to_save_model = op.join(self.working_path, 'models', self.model_name, name + '_model.mdsm')
        self.model.to(torch.device('cpu'))
        torch.save(self.model.state_dict(), path_to_save_model)
        print('Model saved')

    def save_results(self, name=None):
        # Sauvegarde des résultats
        os.makedirs(op.join(self.working_path, 'results'), exist_ok=True)
        if name is None:
            path_to_save_results = op.join(self.working_path, 'results', self.model_name + '_results.json')
        else:
            path_to_save_results = op.join(self.working_path, 'results', name + '_results.json')
        with open(path_to_save_results, 'w') as f:
            json.dump(self.results, f)
        print('Results saved')

    def save_params(self, best_threshold=None, name=None):
        # Sauvegarde des paramètres
        os.makedirs(op.join(self.working_path, 'models'), exist_ok=True)
        if name is not None:
            self.dict_model['model_file'] = op.join(self.working_path, 'models', self.model_name, name + '_model.mdsm')
        else:
            self.dict_model['model_file'] = op.join(self.working_path, 'models', self.model_name + '_model.mdsm')
        self.dict_model['out_channels'] = len(self.sulci_side_list)
        params = {'dict_bck2': self.dict_bck2,
                  'dict_names': self.dict_names,
                  'sulci_side_list': self.sulci_side_list,
                  'dict_model': self.dict_model
                 }
        if best_threshold is not None:
            params['cutting_threshold'] = best_threshold
        if os.path.exists(op.join(self.working_path, 'models', self.model_name )):
            path_to_save_params = op.join(self.working_path, 'models', self.model_name)
        else:
            path_to_save_params = op.join(self.working_path, 'models')
        if name is None:
            path_to_save_params = op.join( path_to_save_params, self.model_name + '_params.json')
        else:
            path_to_save_params = op.join( path_to_save_params, name + '_params.json')
        with open(path_to_save_params, 'w') as f:
            json.dump(params, f)
        print('Parameters saved')

    def reset_results(self):
        self.results = {}

    def load_saved_model(self, dict_model):
        # Chargement d'un modèle précédemment entraîné
        dict_model = self.fill_dict_model(dict_model)

        self.model = UNet3D(dict_model['in_channels'], dict_model['out_channels'],
                               final_sigmoid=dict_model['final_sigmoid'],
                               interpolate=dict_model['interpolate'],
                               conv_layer_order=dict_model['conv_layer_order'],
                               init_channel_number=dict_model['init_channel_number'])
        if dict_model['num_conv'] > 1:
            fac = (dict_model['init_channel_number'] - dict_model['out_channels']) / dict_model['num_conv']
            num_channel = dict_model['init_channel_number']
            self.model.final_conv = nn.Sequential()
            for n in range(self.num_conv):
                self.model.final_conv.add_module(str(n), nn.Conv3d(num_channel - round(n * fac), num_channel  - round((n + 1) * fac), 1))
        else:
            self.model.final_conv = nn.Conv3d(dict_model['init_channel_number'], dict_model['out_channels'],
                                              1)
        self.model.load_state_dict(torch.load(dict_model['model_file'], map_location='cpu'))
        self.model.to(self.device)
        print("Model Loaded !")
