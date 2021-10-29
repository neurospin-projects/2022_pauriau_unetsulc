import json
import random
import os.path as op
from matplotlib import cm
import pandas as pd
import numpy as np

# ----------------------------------------------- #
# Program to create a sulcal_root_colors.hie file
# ----------------------------------------------- #

#Parameters
cohort = 'Chimps3T60'
cohort_eval = 'Chimps3T60TestSet'
model_name = 'RChimps3T60'
hemi = 'R'
learning = 'training'
folds = range(5)

save = True  #to save or note the file
path_to_save = '/host/home/pa267054/data/sulcal_root_color/'  #path where the file is saved
name = 'sulcal_roots_color_bc50_elocal.hie'  #name of the file

alea = False  #colors of each sulcus is random
grad = True  #colors of each sulcus depends on the value of a metric (gradient use : jet)
metric = 'elocal'  #metric from which is compute the color
sbj = 4  # if metric is 'elocal', the subject used to compute the color

# ------------------------------------------------------ #
if op.exists('/host/home/pa267054/Bureau/dico/pauriau/data'):
    path = '/host/home/pa267054/Bureau/dico/pauriau/data'
elif op.exists('/host/nfs/neurospin/dico/pauriau/data'):
    path = '/host/nfs/neurospin/dico/pauriau/data'
else:
    raise FileNotFoundError
# ------------------------------------------------------ #

# Variables
beg = '\n*BEGIN TREE fold_name\nname '
col = '\ncolor '
lab = '\nlabel '
end = '\n\n*END\n'

# File Creation
file = '# tree 1.0\n\n'
file += '*BEGIN TREE hierarchy\ngraph_syntax CorticalFoldArg\n'
file += beg + 'unknown' + col +'255 180 180' + end + beg + 'brain'

if alea:
    for hemi in ['L', 'R']:
        # data import
        cohort_name = 'cohort-' + cohort + '_hemi-' + hemi
        data_file = op.join(path, learning, 'data', cohort_name + '.json')
        data = json.load(open(data_file, 'r'))
        sulci_side_list = data['sulci_side_list']

        # file completion
        file += beg + 'hemisph_' + hemi + '\n'
        file += beg + 'cerebellum_' + hemi + color + '255 0 255' + lab + '20' + end
        for sulci in sulci_side_list:
            if sulci != 'unknown':
                if sulci.startswith('ventricle'):
                    colors = [0, 0, 0]
                else:
                    colors = [random.randint(0, 255) for _ in range(3)]
                file += beg + sulci
                file += col + str(colors[0]) + ' ' + str(colors[1]) + ' ' + str(colors[2])
                #file += lab + str(random.randint(0, 20))
                file += end

        file += end
    file += end
    file += end

# -------------------------------------------------------------------------------------------------------------- #
def create_dico(path, learning, model_name, cohort_eval_name, sulci_side_list, folds=None):
    ''' create a dico with all the evaluation metrics for each sulci'''
    if folds is not None:
        for i, cv in enumerate(folds):
            csv_file = op.join(path, learning, 'evaluations', model_name, model_name + '_cv' + str(cv),
                               cohort_eval_name + '.csv')
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

    dico_eval = {s: {} for s in sulci_side_list}

    for sulci in sulci_side_list:
        dico_eval[sulci]['esi'] = evaluation['ESI_' + sulci].mean()
        dico_eval[sulci]['elocal'] = evaluation['Elocal_' + sulci]
        dico_eval[sulci]['elocal_mean'] = evaluation['Elocal_' + sulci].mean()
        dico_eval[sulci]['elocal_max'] = evaluation['Elocal_' + sulci].max()

    return dico_eval
# -------------------------------------------------------------------------------------------------------------- #

if grad:
    # data import
    cohort_name = 'cohort-' + cohort + '_hemi-' + hemi
    cohort_eval_name = 'cohort-' + cohort_eval + '_hemi-' + hemi
    data_file = op.join(path, learning, 'data', cohort_name + '.json')
    data = json.load(open(data_file, 'r'))
    sulci_side_list = data['sulci_side_list']

    #dico import
    dico = create_dico(path, learning, model_name, cohort_eval_name, sulci_side_list, folds)

    # file completion
    file += beg + 'hemisph_' + hemi + '\n'
    file += beg + 'cerebellum_' + hemi + col + '255 0 255' + lab + '20' + end
    for sulci in sulci_side_list:
        if sulci != 'unknown':
            if sulci.startswith('ventricle'):
                colors = [0, 0, 0]
            else:
                if metric == 'elocal':
                    colors = [int(255*cm.jet(dico[sulci][metric][sbj])[i]) for i in range(3)]
                else:
                    colors = [int(255*cm.jet(dico[sulci][metric])[i]) for i in range(3)]
            file += beg + sulci
            file += col + str(colors[0]) + ' ' + str(colors[1]) + ' ' + str(colors[2])
            # file += lab + str(random.randint(0, 20))
            file += end

    file += end
    file += end
    file += end

if save:
    with open(op.join(path_to_save, name), 'w') as f:
        f.write(file)