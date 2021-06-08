import json
import numpy as np
import copy
import time
import os
import sigraph

import torch
import torch.nn as nn
import torch.optim as optim

from soma import aims

#from deepsulci.deeptools import dataset
#from deepsulci.sulci_labeling.method.unet import UnetSulciLabeling
from deepsulci.deeptools.dataset import extract_data, SulciDataset
from deepsulci.deeptools.models import UNet3D
from deepsulci.sulci_labeling.analyse.stats import esi_score
from sklearn.model_selection import train_test_split

# DATA INPUT
cohort_name = 'Chimp_3T'
hemi = 'L'
path_to_cohort = '/volatile/home/pierre/Data/cohorts'

with open(os.path.join(path_to_cohort, 'cohort-'+cohort_name+'_hemi-'+hemi+'.json'), 'r') as f:
    cohort = json.load(f)

graphs = []
for s in cohort['subjects']:
    graphs.append(s['graph'])
graphs = graphs[:10]
translation_file = '/casa/host/build/share/brainvisa-share-5.1/nomenclature/translation/sulci_model_2018.trl'

# SULCI SIDE LIST
print('Creating sulci side list...')
if os.path.exists(translation_file):
    flt = sigraph.FoldLabelsTranslator()
    flt.readLabels(translation_file)
    trfile = translation_file
else:
    trfile = None
    print('Translation file not found.')

sulci_side_list = set()
dict_bck2, dict_names = {}, {}
for gfile in graphs:
    graph = aims.read(gfile)
    if trfile is not None:
        flt.translate(graph)
    data = extract_data(graph)
    dict_bck2[gfile] = data['bck2']
    dict_names[gfile] = data['names']
    for n in data['names']:
        sulci_side_list.add(n)
sulci_side_list = sorted(list(sulci_side_list))
dict_sulci = {sulci_side_list[i]: i for i in range(len(sulci_side_list))}
if 'background' not in dict_sulci:
    dict_sulci['background'] = -1
sslist = [ss for ss in sulci_side_list if not ss.startswith('unknown') and not ss.startswith('ventricle')]
background = dict_sulci['background']


# # DATASET / DATALOADERS
print('Extract validation dataloader...')
batch_size = 1

gfile_list_train, gfile_list_test = train_test_split(graphs, test_size=0.1)
valdataset = SulciDataset(
    gfile_list_test, dict_sulci,
    train=False, translation_file=translation_file,
    dict_bck2=dict_bck2, dict_names=dict_names)
valloader = torch.utils.data.DataLoader(
    valdataset, batch_size=batch_size,
    shuffle=False, num_workers=0)

print('Extract train dataloader...')
traindataset = SulciDataset(
    gfile_list_train, dict_sulci,
    train=True, translation_file=translation_file,
    dict_bck2=dict_bck2, dict_names=dict_names)
trainloader = torch.utils.data.DataLoader(
    traindataset, batch_size=batch_size,
    shuffle=False, num_workers=0)


# MODEL
# Load file
print('Network initialization...')
num_channel = 1
num_filter = 64
lr = 1e-5
momentum = 1

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


if hemi == 'L':
    model_file = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_left.mdsm'
    with open('/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_params_left.json', 'r') as f:
        param = json.load(f)
else:
    model_file = '/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_right.mdsm'
    with open('/casa/host/build/share/brainvisa-share-5.1/models/models_2019/cnn_models/sulci_unet_model_params_right.json', 'r') as f:
        param = json.load(f)
print(param.keys())

trained_sulci_side_list = param['sulci_side_list']
trained_model = UNet3D(num_channel, len(trained_sulci_side_list), final_sigmoid=False, init_channel_number=num_filter)
trained_model.load_state_dict(torch.load(model_file, map_location='cpu'))
# for p in trained_model.state_dict():
#     print(p, "\t", trained_model.state_dict()[p].size())
model = copy.deepcopy(trained_model)
model.final_conv = nn.Conv3d(num_filter, len(sulci_side_list), 1)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=0)
criterion = nn.CrossEntropyLoss(ignore_index=-1)


# # TRAINING
print('training...')
num_epochs = 5
path_to_save_model = '/volatile/home/pierre/Data/model'

since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc, epoch_acc = 0., 0.
best_epoch = 0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    start_time = time.time()

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()    # Set model to evaluate mode

        running_loss = 0.0

        # compute dataloader
        dataloader = trainloader if phase == 'train' else valloader

        # Iterate over data.
        y_pred, y_true = [], []
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forwards.det
            # track history if only in train
            # with torch.set_grad_enabled(phase == 'train'):
            #     outputs = model(inputs)
            #     _, preds = torch.max(outputs, 1)
            #     loss = criterion(outputs, labels)
            #
            #     # backward + optimize only if in training phase
            #     if phase == 'train':
            #         loss.backward()
            #         optimizer.step()
            if phase == 'train':
                for name, parameters in model.named_parameters():
                    if name == 'final_conv.weight' or name == 'final_conv.bias':
                        parameters.requires_grad = True
                    else:
                        parameters.requires_grad = False
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                 with torch.set_grad_enabled(phase == 'train'):
                     outputs = model(inputs)
                     _, preds = torch.max(outputs, 1)
                     loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            y_pred.extend(preds[labels != background].tolist())
            y_true.extend(labels[labels != background].tolist())

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = 1 - esi_score(
            y_true, y_pred,
            [dict_sulci[ss] for ss in sslist])

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
#
#     # early_stopping
#     es_stop(epoch_loss, model)
#     divide_lr(epoch_loss, model)
#
#     if divide_lr.early_stop:
#         print('Divide learning rate')
#         lr = lr/2
#         optimizer = optim.SGD(model.parameters(), lr=lr,
#                               momentum=self.momentum)
#         divide_lr = EarlyStopping(patience=patience)
#
#     if es_stop.early_stop:
#         print("Early stopping")
#         break
#
    print('Epoch took %i s.' % (time.time() - start_time))
    print('\n')

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}, Epoch {}'.format(best_acc, best_epoch))

# load best model weights and save it !
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), path_to_save_model)