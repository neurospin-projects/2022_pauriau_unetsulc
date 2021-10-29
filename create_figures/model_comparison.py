import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path as op
from scipy.stats import wilcoxon

# ----------------------------------------------------------------------
# Script qui permet de tracer les courbes d'évaluation pour comparer
# les performances de 2 modèles entraînés et évalués sur la même cohorte
# ----------------------------------------------------------------------


#Parameters
#models
model_name_1 = 'RChimps3T60'
label_1 = 'cropped' #label du model sur les figures
learning_1 = 'training' #méthode d'entraînement : 'training' ou 'transfer_learning"

model_name_2 = 'RChimps3T60BI'
label_2 = 'set'
learning_2 = 'training'

folds = 5 #nombre de folds
num_epochs = 100 #nombre d"époques pour l'entraînement (pour les figures 1 et 2)

#cohorte
full_cohort = 'Chimps3T' #nom de la cohorte qui contient l'ensemble des graphes
cohort_train = 'Chimps3T60' #nom de la cohorte pour l'entraînement
cohort_eval = 'Chimps3T60TestSet' #nom de la cohorte pour l'évaluation des modèles
hemi = 'R'
remove = True #Retirer ou non les sillons 'ventricle' et 'unknown" de la liste des sillons

#figures
save = False #sauvegarde des figures
figures = [1, 2, 3, 4, 5, 6] #liste des figures à afficher
path_to_save = op.join('figures') #dossier de sauvegarde des images
legend_title = 'Image Size:'

# --- Data Import --- #
full_cohort_name = 'cohort-' + full_cohort + '_hemi-' + hemi
cohort_name = 'cohort-' + cohort_train + '_hemi-' + hemi
cohort_eval_name = 'cohort-' + cohort_eval + '_hemi-' + hemi


if op.exists('/host/nfs/neurospin/dico/pauriau/data'):
    path = '/host/nfs/neurospin/dico/pauriau/data'
elif op.exists('/host/neurospin/dico/pauriau/data'):
    path = '/host/neurospin/dico/pauriau/data'
elif op.exists('/host/home/pa267054/Bureau/dico/pauriau/data'):
    path = '/host/home/pa267054/Bureau/dico/pauriau/data'
else:
    raise FileNotFoundError


### COHORT FILE ###
#Sulci side list
data_file = op.join(path, learning_1, 'data', full_cohort_name + '.json')
data = json.load(open(data_file, 'r'))
sulci_side_list = data['sulci_side_list']
dict_names = data['dict_names']
print('Nombre de sillon dans la cohorte : ', len(sulci_side_list))

dico_cohort = {s: {} for s in sulci_side_list}

for sulci in sulci_side_list:
    dico_cohort[sulci]['sizes'] = [n.count(sulci) for n in dict_names.values() if sulci in n]
    dico_cohort[sulci]['occurrences'] = np.sum([1 for n in dict_names.values() if sulci in n])

paramfile = op.join(path, learning_1, 'models', model_name_1, model_name_1 + '_cv0_params.json')
param = json.load(open(paramfile, 'r'))
sulci_side_list_train = param['sulci_side_list']
print("Nombre de sillon dans la cohorte d'entraînement : ", len(sulci_side_list_train))

sorted_sulci_side_list = sorted(sulci_side_list_train, key=lambda x: np.mean(dico_cohort[x]['sizes']), reverse=True)
if remove:
    sorted_sulci_side_list.remove('unknown')
    for s in sorted_sulci_side_list:
        if s.startswith('ventricle'):
            sorted_sulci_side_list.remove(s)

### CSV FILE ###

for cv in range(folds):
    csv_file_1 = op.join(path, learning_1, 'evaluations', model_name_1, model_name_1 + '_cv' + str(cv), cohort_eval_name + '.csv')
    csv_file_2 = op.join(path, learning_2, 'evaluations', model_name_2, model_name_2 + '_cv' + str(cv), cohort_eval_name + '.csv')

    evaluation_1 = pd.read_csv(csv_file_1)
    evaluation_2 = pd.read_csv(csv_file_2)

    if cv == 0:
        big_evaluation_1 = evaluation_1.copy()
        big_evaluation_2 = evaluation_2.copy()
    else:
        big_evaluation_1 = pd.concat((big_evaluation_1, evaluation_1))
        big_evaluation_2 = pd.concat((big_evaluation_2, evaluation_2))


by_row_index_1 = big_evaluation_1.groupby(big_evaluation_1.index)
evaluation_means_1 = by_row_index_1.mean()
by_row_index_2 = big_evaluation_2.groupby(big_evaluation_2.index)
evaluation_means_2 = by_row_index_2.mean()

#DICO STATS

dico_stats_1 = {s: {} for s in sorted_sulci_side_list}
dico_stats_2 = {s: {} for s in sorted_sulci_side_list}

for sulci in sorted_sulci_side_list:
    dico_stats_1[sulci]['accuracy'] = evaluation_means_1['acc_' + sulci].mean()
    dico_stats_1[sulci]['sensitivity'] = evaluation_means_1['sens_' + sulci].mean()
    dico_stats_1[sulci]['specificity'] = evaluation_means_1['spec_' + sulci].mean()
    dico_stats_1[sulci]['balanced_accuracy'] = evaluation_means_1['bacc_' + sulci].mean()
    dico_stats_1[sulci]['esi'] = evaluation_means_1['ESI_' + sulci].mean()
    dico_stats_1[sulci]['elocal'] = list(evaluation_means_1['Elocal_' + sulci])
    dico_stats_1[sulci]['elocal_mean'] = evaluation_means_1['Elocal_' + sulci].mean()
    dico_stats_1[sulci]['elocal_max'] = evaluation_means_1['Elocal_' + sulci].max()

    dico_stats_2[sulci]['accuracy'] = evaluation_means_2['acc_' + sulci].mean()
    dico_stats_2[sulci]['sensitivity'] = evaluation_means_2['sens_' + sulci].mean()
    dico_stats_2[sulci]['specificity'] = evaluation_means_2['spec_' + sulci].mean()
    dico_stats_2[sulci]['balanced_accuracy'] = evaluation_means_2['bacc_' + sulci].mean()
    dico_stats_2[sulci]['esi'] = evaluation_means_2['ESI_' + sulci].mean()
    dico_stats_2[sulci]['elocal'] = list(evaluation_means_2['Elocal_' + sulci])
    dico_stats_2[sulci]['elocal_mean'] = evaluation_means_2['Elocal_' + sulci].mean()
    dico_stats_2[sulci]['elocal_max'] = evaluation_means_2['Elocal_' + sulci].max()

### RESULT FILE ###
result_file_1 = op.join(path, learning_1, 'results', model_name_1 + '.json')
result_1 = json.load(open(result_file_1, 'r'))
result_file_2 = op.join(path, learning_2, 'results', model_name_2 + '.json')
result_2 = json.load(open(result_file_2, 'r'))

dico_results_1 = {
'loss_val' : [],
'loss_train' : [],
'acc_train' : [],
'acc_val' : []
}
dico_results_2 = {
'loss_val' : [],
'loss_train' : [],
'acc_train' : [],
'acc_val' : []
}

for i in range(folds):
    if len(result_1['epoch_loss_train'][i]) < num_epochs:
        dico_results_1['loss_train'].append(result_1['epoch_loss_train'][i] + [result_1['epoch_loss_train'][i][-1] for j in range(num_epochs - len(result_1['epoch_loss_train'][i]))])
        dico_results_1['loss_val'].append(result_1['epoch_loss_val'][i] + [result_1['epoch_loss_val'][i][-1] for j in range(num_epochs - len(result_1['epoch_loss_val'][i]))])
        dico_results_1['acc_train'].append(result_1['epoch_acc_train'][i] + [result_1['epoch_acc_train'][i][-1] for j in range(num_epochs - len(result_1['epoch_acc_train'][i]))])
        dico_results_1['acc_val'].append(result_1['epoch_acc_val'][i] + [result_1['epoch_acc_val'][i][-1] for j in range(num_epochs - len(result_1['epoch_acc_val'][i]))])
    else:
        dico_results_1['loss_train'].append(result_1['epoch_loss_train'][i])
        dico_results_1['loss_val'].append(result_1['epoch_loss_val'][i])
        dico_results_1['acc_train'].append(result_1['epoch_acc_train'][i])
        dico_results_1['acc_val'].append(result_1['epoch_acc_val'][i])

    if len(result_2['epoch_loss_train'][i]) < num_epochs:
        dico_results_2['loss_train'].append(result_2['epoch_loss_train'][i] + [result_2['epoch_loss_train'][i][-1] for j in range(num_epochs - len(result_2['epoch_loss_train'][i]))])
        dico_results_2['loss_val'].append(result_2['epoch_loss_val'][i] + [result_2['epoch_loss_val'][i][-1] for j in range(num_epochs - len(result_2['epoch_loss_val'][i]))])
        dico_results_2['acc_train'].append(result_2['epoch_acc_train'][i] + [result_2['epoch_acc_train'][i][-1] for j in range(num_epochs - len(result_2['epoch_acc_train'][i]))])
        dico_results_2['acc_val'].append(result_2['epoch_acc_val'][i] + [result_2['epoch_acc_val'][i][-1] for j in range(num_epochs - len(result_2['epoch_acc_val'][i]))])
    else:
        dico_results_2['loss_train'].append(result_2['epoch_loss_train'][i][:num_epochs])
        dico_results_2['loss_val'].append(result_2['epoch_loss_val'][i][:num_epochs])
        dico_results_2['acc_train'].append(result_2['epoch_acc_train'][i][:num_epochs])
        dico_results_2['acc_val'].append(result_2['epoch_acc_val'][i][:num_epochs])

dico_results_1['acc_train_mean'] = np.mean(dico_results_1['acc_train'], axis=0)
dico_results_1['loss_train_mean'] = np.mean(dico_results_1['loss_train'], axis=0)
dico_results_1['acc_val_mean'] = np.mean(dico_results_1['acc_val'], axis=0)
dico_results_1['loss_val_mean'] = np.mean(dico_results_1['loss_val'], axis=0)

dico_results_2['acc_train_mean'] = np.mean(dico_results_2['acc_train'], axis=0)
dico_results_2['loss_train_mean'] = np.mean(dico_results_2['loss_train'], axis=0)
dico_results_2['acc_val_mean'] = np.mean(dico_results_2['acc_val'], axis=0)
dico_results_2['loss_val_mean'] = np.mean(dico_results_2['loss_val'], axis=0)


# --- Figures --- #
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('legend', fontsize='x-small', title_fontsize='x-small')
plt.rc('figure', titlesize='small')

colors = ['blue', 'orange', 'green', 'red', 'purple']

if hemi == 'R':
    n = 6
else:
    n = 5

if 1 in figures:
    plt.figure(1, figsize=(4, 4))
    #plt.title('Model: ' + model_name_1 + '/' + model_name_2 + '\nCohort: ' + cohort_eval + ' | Hemi: ' + hemi, fontsize='small')
    plt.plot(dico_results_1['loss_train_mean'], linestyle='--', color=colors[0])
    plt.plot(dico_results_1['loss_val_mean'], linestyle='-', color=colors[0])
    plt.plot(dico_results_2['loss_train_mean'], linestyle='--', color=colors[1])
    plt.plot(dico_results_2['loss_val_mean'], linestyle='-', color=colors[1])
    a, = plt.plot([0], [0], color='black', linestyle='--', label='train set')
    b, = plt.plot([0], [0], color='black', linestyle='-', label='val set')
    c, = plt.plot([0], [0], color=colors[0], label=label_1)
    d, = plt.plot([0], [0], color=colors[1], label=label_2)
    plt.legend(handles=[a, b, c, d])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplots_adjust(left=0.125, right=0.95, bottom=0.15, top=0.95)
    if save:
        plt.savefig(op.join(path_to_save, model_name_1 + '_' + model_name_2 + '_loss.png'))

if 2 in figures:
    plt.figure(2, figsize=(4, 4))
    #plt.title('Model: ' + model_name_1 + '/' + model_name_2 + '\nCohort: ' + cohort_eval + ' | Hemi: ' + hemi, fontsize='small')
    plt.plot(dico_results_1['acc_train_mean'], linestyle='--', color=colors[0])
    plt.plot(dico_results_1['acc_val_mean'], color=colors[0])
    plt.plot(dico_results_2['acc_train_mean'], linestyle='--', color=colors[1])
    plt.plot(dico_results_2['acc_val_mean'], color=colors[1])
    a, = plt.plot([0], [0], color='black', linestyle='--', label='train set')
    b, = plt.plot([0], [0], color='black', linestyle='-', label='val set')
    c, = plt.plot([0], [0], color=colors[0], label=label_1)
    d, = plt.plot([0], [0], color=colors[1], label=label_2)
    plt.legend(handles=[a, b, c, d])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(left=0.125, right=0.95, bottom=0.15, top=0.95)
    if save:
        plt.savefig(op.join(path_to_save, model_name_1 + '_' + model_name_2 +'_acc.png'))

if 3 in figures:
    plt.figure(3, figsize=(4, 6))
    #plt.title('Model: ' + model_name_1 + '/' + model_name_2 + '\nCohort: ' + cohort_eval + ' | Hemi: ' + hemi, fontsize='small')
    plt.barh(y=range(len(sorted_sulci_side_list)), width=[dico_stats_1[sulci]['elocal_mean'] for sulci in sorted_sulci_side_list], height=0.8, align='center', alpha=0.5, label=label_1)
    plt.barh(y=range(len(sorted_sulci_side_list)), width=[dico_stats_2[sulci]['elocal_mean'] for sulci in sorted_sulci_side_list], height=0.8, align='center', alpha=0.5, label=label_2)
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.xlabel('$E_{local}^{mean}$')
    plt.legend(loc='upper right', title=legend_title)
    plt.subplots_adjust(left=0.25, right=0.92, bottom=0.08, top=0.95)
    if save:
        plt.savefig(op.join(path_to_save, model_name_1 + '_' + model_name_2 + '_elocal.png'))

if 4 in figures:
    plt.figure(4, figsize=(4, 6))
    #plt.title('Model: ' + model_name_1 + '/' + model_name_2 + '\nCohort: ' + cohort_eval + ' | Hemi: ' + hemi, fontsize='small')
    plt.barh(y=range(len(sorted_sulci_side_list)), width=[dico_stats_1[sulci]['elocal_max'] for sulci in sorted_sulci_side_list], height=0.8, align='center', alpha=0.5, label=label_1)
    plt.barh(y=range(len(sorted_sulci_side_list)), width=[dico_stats_2[sulci]['elocal_max'] for sulci in sorted_sulci_side_list], height=0.8, align='center', alpha=0.5, label=label_2)
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.xlabel('$E_{local}^{max}$')
    plt.legend(loc='lower right', title=legend_title)
    plt.subplots_adjust(left=0.25, right=0.92, bottom=0.08, top=0.95)
    if save:
        plt.savefig(op.join(path_to_save, model_name_1 + '_' + model_name_2 + '_elocal_max.png'))

if 5 in figures:
    plt.figure(5, figsize=(4, 6))
    #plt.title('Model: ' + model_name_1 + '/' + model_name_2 + '\nCohort: ' + cohort_eval + ' | Hemi: ' + hemi, fontsize='small')
    ect_1 = [np.std(dico_stats_1[sulci]['elocal']) for sulci in sorted_sulci_side_list]
    ect_2 = [np.std(dico_stats_2[sulci]['elocal']) for sulci in sorted_sulci_side_list]
    mean_1 = [np.mean(dico_stats_1[sulci]['elocal']) for sulci in sorted_sulci_side_list]
    mean_2 = [np.mean(dico_stats_2[sulci]['elocal']) for sulci in sorted_sulci_side_list]
    plt.barh(y=range(len(sorted_sulci_side_list)), width=mean_1,
             xerr=ect_1, ecolor=colors[0], capsize=1.5, height=0.5, align='center', alpha=0.5, label=label_1)
    plt.barh(y=[0.3 + i for i in range(len(sorted_sulci_side_list))], width=mean_2,
             xerr=ect_2, ecolor=colors[1], capsize=1.5, height=0.5, align='center', alpha=0.5, label=label_2)
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.xlabel('$E_{local}^{mean}$')
    plt.legend(loc='lower right', title=legend_title)
    plt.subplots_adjust(left=0.25, right=0.92, bottom=0.08, top=0.95)
    if save:
        plt.savefig(op.join(path_to_save, model_name_1 + '_' + model_name_2 + '_elocal_mean_ect.png'))

if 6 in figures:
    plt.figure(6, figsize=(4, 6))
    plt.boxplot([dico_stats_1[sulci]['elocal'] for sulci in sorted_sulci_side_list], vert=False,
                showfliers=False, notch=False, showmeans=False, meanprops={}, medianprops={'color': 'red', 'linewidth': 2},
                boxprops={'color': 'blue'}, capprops={'color': 'blue'}, whiskerprops={'color': 'blue'},
                positions=[i-0.5 for i in range(1, 2*(len(sorted_sulci_side_list)), 2)])
    plt.boxplot([dico_stats_2[sulci]['elocal'] for sulci in sorted_sulci_side_list], vert=False,
                showfliers=False, notch=False, showmeans=False, meanprops={}, medianprops={'color': 'green', 'linewidth': 2},
                boxprops={'color': 'orange'}, capprops={'color': 'orange'}, whiskerprops={'color': 'orange'},
                positions=[i+0.5 for i in range(1, 2*(len(sorted_sulci_side_list)), 2)])
    plt.yticks(range(1, 2*(len(sorted_sulci_side_list)), 2), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.subplots_adjust(left=0.25, right=0.92, bottom=0.08, top=0.95)

plt.show()