import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path as op

# ----------------------------------------------------------------------
# Script qui permet de tracer les figures des résultats d'un entraînement
# d'un modèle.
# ----------------------------------------------------------------------


#Parameters
model_name = 'RChimps3T60'
learning = 'training'  #training or transfer_learning
folds = 5  #number of folds

cohort = 'Chimps3T'  #full cohort
cohort_train = 'Chimps3T60'
cohort_eval = 'Chimps3T60TestSet'
hemi = 'R'
remove = True  #remove or not ventricle and unknown

path = '/host/nfs/neurospin/dico/pauriau/data' #working path

figures = [1, 2, 3, 4, 5]
save = False
path_to_save_figures = ''


# --- Data Import --- #

cohort_name = 'cohort-' + cohort + '_hemi-' + hemi
cohort_train_name = 'cohort-' + cohort_train + '_hemi-' + hemi
cohort_eval_name = 'cohort-' + cohort_eval + '_hemi-' + hemi

#csv file
if folds is not None:
    for i, cv in enumerate(range(folds)):
        csv_file = op.join(path, learning, 'evaluations', model_name, model_name + '_cv' + str(cv), cohort_eval_name + '.csv')
        eval = pd.read_csv(csv_file)
        if i == 0:
            big_eval = eval.copy()
        else:
            big_eval = pd.concat((big_eval, eval))
    by_row_index = big_eval.groupby(big_eval.index)
    evaluation = by_row_index.mean()
else:
    csv_file = op.join(path, learning, 'evaluations', model_name, cohort_eval_name + '.csv')
    evaluation = pd.read_csv(csv_file)


#data file
data_file = op.join(path, learning, 'data', cohort_train_name + '.json')
data = json.load(open(data_file, 'r'))
#result file
result_file = op.join(path, learning, 'results', model_name + '.json')
result = json.load(open(result_file, 'r'))
#cohort file
cohort_file = op.join(path, learning, 'data', cohort_name + '.json')
cohort_data = json.load(open(cohort_file, 'r'))

# --- Compute Results --- #
### DATA FILE ###
sulci_side_list = data['sulci_side_list']

### COHORT FILE ###
full_sulci_side_list = cohort_data['sulci_side_list']
full_dict_names = cohort_data['dict_names']

dico_cohort = {s: {} for s in full_sulci_side_list}

for sulci in full_sulci_side_list:
    dico_cohort[sulci]['sizes'] = [n.count(sulci) for n in full_dict_names.values() if sulci in n]
    dico_cohort[sulci]['occurrences'] = np.sum([1 for n in full_dict_names.values() if sulci in n])

sorted_sulci_side_list = sorted(sulci_side_list, key=lambda x: np.mean(dico_cohort[x]['sizes']), reverse=True)

if remove:
    sorted_sulci_side_list.remove('unknown')
    for s in sorted_sulci_side_list:
        if s.startswith('ventricle'):
            sorted_sulci_side_list.remove(s)

### CSV FILE ###
dico_eval = {s: {} for s in sulci_side_list}
for sulci in sulci_side_list:
    dico_eval[sulci]['accuracy'] = evaluation['acc_' + sulci].mean()
    dico_eval[sulci]['sensitivity'] = evaluation['sens_' + sulci].mean()
    dico_eval[sulci]['specificity'] = evaluation['spec_' + sulci].mean()
    dico_eval[sulci]['balanced_accuracy'] = evaluation['bacc_' + sulci].mean()
    dico_eval[sulci]['esi'] = evaluation['ESI_' + sulci].mean()
    dico_eval[sulci]['elocal'] = evaluation['Elocal_' + sulci]
    dico_eval[sulci]['elocal_mean'] = evaluation['Elocal_' + sulci].mean()
    dico_eval[sulci]['elocal_max'] = evaluation['Elocal_' + sulci].max()
    dico_eval[sulci]['iou'] = (evaluation['TP_' + sulci] / (evaluation['TP_' + sulci] + evaluation['FN_' + sulci] + evaluation['FP_' + sulci])).mean()

### RESULT FILE ###
epoch_loss_train = result['epoch_loss_train']
epoch_loss_val = result['epoch_loss_val']
epoch_acc_train = result['epoch_acc_train']
epoch_acc_val = result['epoch_acc_val']

best_epoch = result['best_epoch']
best_acc = result['best_acc']


# ---- Print Results ------ #
print('Nombre de sillons dans la cohorte : ', len(full_sulci_side_list))
print("Nombre de sillons dans la cohorte d'entraînement : ", len(sulci_side_list))

print('Average ESI on : ')
print('\t-Train Set: ', 1 - np.mean([np.max(result['epoch_acc_train'][i]) for i in range(folds)]))
print('\t-Validation Set : ', 1 - np.mean(result['best_acc']))
print('\t-Test Set : ', np.mean(evaluation['ESI']))
print('Average best epoch : ', np.mean(result['best_epoch']))

# --- Figures --- #
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('legend', fontsize='x-small', title_fontsize='x-small')
plt.rc('figure', titlesize='small')
plt.rc('axes', titlesize='small')
colors = ['blue', 'orange', 'green', 'red', 'purple']
if hemi == 'R':
    n = 6
elif hemi == 'L':
    n = 5

#function to add label at the end to the bar
def addlabels(y, x, data):
    for i in range(len(x)):
        if not np.isnan(data[i]):
            color = 'black'
            if x[i] > np.max(x) - 0.1:
                plt.text(x[i] + 0.01, y[i] - 0.3, round(data[i], 3), ha='left', fontsize='xx-small', color=color,
                         bbox=dict(boxstyle="square, pad=0.5", color='white'))
            else:
                plt.text(x[i] + 0.01, y[i] - 0.3, round(data[i], 3), ha='left', fontsize='xx-small', color=color)


if 1 in figures:
    plt.figure(1, figsize=(4, 4))
    plt.title('Training of Model:' + model_name + '\nOn Cohort:' + cohort_eval + ' | Hemi: ' + hemi)
    for cv in range(len(epoch_loss_train)):
        plt.plot(epoch_loss_train[cv], linestyle='--', alpha=0.7, color=colors[cv])
        plt.plot(epoch_loss_val[cv], linestyle='-', alpha=0.7, color=colors[cv])
    a, = plt.plot([0], [0], color='black', linestyle='--', label='train set')
    b, = plt.plot([0], [0], color='black', linestyle='-', label='val set')
    plt.legend(handles=[a, b])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplots_adjust(left=0.125, right=0.95, bottom=0.12, top=0.92)
    if save:
        plt.savefig(op.join(path_to_save_figures, model_name + '_loss.png'))

if 2 in figures:
    plt.figure(2, figsize=(4, 4))
    plt.title('Training of Model:' + model_name + '\nOn Cohort:' + cohort_eval + ' | Hemi: ' + hemi)
    for cv in range(len(epoch_loss_train)):
        plt.plot(epoch_acc_train[cv], linestyle='--', alpha=0.8, color=colors[cv])
        plt.plot(epoch_acc_val[cv], linestyle='-', alpha=0.7, color=colors[cv])
    a, = plt.plot([0], [0], color='black', linestyle='--', label='train set')
    b, = plt.plot([0], [0], color='black', linestyle='-', label='val set')
    plt.legend(handles=[a, b])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(left=0.125, right=0.95, bottom=0.12, top=0.92)
    if save:
        plt.savefig(op.join(path_to_save_figures, model_name + '_acc.png'))

if 3 in figures:
    plt.figure(3, figsize=(4, 6))
    plt.title('Evaluation of Model: ' + model_name + '\nOn Cohort:' + cohort_eval + '/ Hemi: ' + hemi)
    plt.barh(y=range(len(sorted_sulci_side_list)), width=[dico_eval[s]['elocal_mean'] for s in sorted_sulci_side_list], height=0.8, align='center')
    xerr = [np.std(dico_eval[s]['elocal']) for s in sorted_sulci_side_list]
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.xlabel('$E_{local}^{mean}$')
    addlabels(range(len(sorted_sulci_side_list)), [dico_eval[s]['elocal_mean'] for s in sorted_sulci_side_list], xerr)
    plt.subplots_adjust(left=0.25, right=0.96, bottom=0.07, top=0.92)
    if save:
        plt.savefig(op.join(path_to_save_figures, model_name + '_elocal_mean.png'))

if 4 in figures:
    plt.figure(4, figsize=(4, 6))
    plt.title('Evaluation of Model: ' + model_name + '\nOn Cohort:' + cohort_eval + '/ Hemi: ' + hemi)
    plt.barh(y=range(len(sorted_sulci_side_list)), width=[dico_eval[s]['elocal_max'] for s in sorted_sulci_side_list], height=0.8, align='center')
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.subplots_adjust(left=0.28, right=0.92, bottom=0.08, top=0.92)
    plt.xlabel('$E_{local}^{max}$')
    if save:
        plt.savefig(op.join(path_to_save_figures, model_name + '_elocal_max.png'))

if 5 in figures:
    plt.figure(5, figsize=(4, 6))
    plt.title('Evaluation of Model: ' + model_name + '\nOn Cohort:' + cohort_eval + '/ Hemi: ' + hemi)
    plt.boxplot([dico_eval[sulci]['elocal'] for sulci in sorted_sulci_side_list], vert=False,
                showfliers=False, notch=False, showmeans=False, meanprops={}, medianprops={'color': 'orange', 'linewidth': 2},
                boxprops={'color': 'blue'}, capprops={'color': 'blue'}, whiskerprops={'color': 'blue'},
                positions=range(len(sorted_sulci_side_list)))
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.xlabel('$E_{local}$')
    plt.subplots_adjust(left=0.25, right=0.98, bottom=0.07, top=0.92)
    if save:
        plt.savefig(op.join(path_to_save_figures, model_name + '_elocal.png'))

plt.show()
