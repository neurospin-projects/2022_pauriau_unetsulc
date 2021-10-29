import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path as op
import sigraph
from deepsulci.deeptools.dataset import extract_data
from soma import aims

# -----------------------------------------------------------------
# Program to plot sizes and occurrences of the sulcus in the cohort
# -----------------------------------------------------------------

#Parameters
cohort = 'Chimps3T'
hemi = 'R'
path = '/host/home/pa267054/data/cohorts' #path to the cohort file

#data_file
data_file = '/host/home/pa267054/data/tests/data'  # None or the path to the data file
remove = True  #remove or not 'unknown' from sulci side list
if data_file is None:
    save = True  # Save or not the data file
    path_to_save = ''  #path to save the data file

#figures
savefig = False
path_to_save_figures = ''
figures = [1, 2, 3, 4]  # List of figures to plot

# --- Data Import --- #
    
cohort_name = 'cohort-' + cohort + '_hemi-' + hemi
cohort_file = op.join(path, cohort_name + '.json')
cohort_dict = json.load(open(cohort_file, 'r'))

# --------------------------------------------------------- #
def extract_data_from_graphs(graphs, translation_file=None):
    '''function to create sulci_side_list, dict_names,... from graphs'''
    if translation_file is not None:
        flt = sigraph.FoldLabelsTranslator()
        flt.readLabels(translation_file)

    print('Creating sulci side list...')
    sulci_side_list = set()
    dict_bck2, dict_names = {}, {}
    for gfile in graphs:
        graph = aims.read(gfile)
        if translation_file is not None:
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
    dict_bck2 = dict_bck2
    return sulci_side_list, sslist, dict_bck2, dict_sulci, dict_names
# ------------------------------------------------------------------ #

#Create data file
if data_file is not None:
    data = json.load(open(op.join(data_file, cohort_name + '.json'), 'r'))
    sulci_side_list = data['sulci_side_list']
    dict_names = data['dict_names']
else:
    graphs = [s['graph'] for s in cohort_dict['subjects']]
    sulci_side_list, sslist, dict_bck2, dict_sulci, dict_names = extract_data_from_graphs(graphs)
    if save:
        pts = op.join(path_to_save, 'data', cohort_name + '.json')
        data = {
            'sulci_side_list': sulci_side_list,
            'dict_bck2': dict_bck2,
            'dict_names': dict_names
        }
        with open(pts, 'w') as f:
            json.dump(data, f)

#Create dico stats
#Sizes and occurrences of sulci in the cohort
#Create sorted sulci_side_list
dico_stats = {sulci: {} for sulci in sulci_side_list}
for sulci in sulci_side_list:
    nb_occ = 0
    sizes = []
    dico_stats[sulci]['graphs'] = []
    for graph, names in dict_names.items():
        if sulci in names:
            nb_occ += 1
            dico_stats[sulci]['graphs'].append(graph)
            sizes.append(names.count(sulci))
    dico_stats[sulci]['occurrences'] = nb_occ
    dico_stats[sulci]['sizes'] = sizes

sorted_sulci_side_list = sorted(sulci_side_list, key=lambda x: np.mean(dico_stats[x]['sizes']), reverse=True)

print('Cohort : ', cohort_dict['name'])
print('Nombre de sujets : ', len(cohort_dict['subjects']))
print('Nombre de sillons : ', len(sulci_side_list))


# ----------- Figures --------------- #
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('legend', fontsize='x-small', title_fontsize='x-small')
plt.rc('figure', titlesize='small')
if hemi == 'R':
    n = 6
elif hemi == 'L':
    n = 5

#Parameters to correct the borders of the plots
left = 0.24
right = 0.92
bottom = 0.08
top = 0.92

if 1 in figures:
    plt.figure(1, figsize=(3, 6))
    plt.title('Cohort: ' + cohort + ' | Hemi: ' + hemi)
    plt.barh(y=range(len(sorted_sulci_side_list)), width=[dico_stats[sulci]['occurrences'] for sulci in sorted_sulci_side_list], height=0.8, align='center', label='occurrences')
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.xlabel('Number of graphs')
    plt.legend(loc='upper right')
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    if savefig:
        plt.savefig(op.join(path_to_save_figures, cohort_name + '_occ.png'))

if 2 in figures:
    plt.figure(2, figsize=(3, 6))
    plt.title('Cohort: ' + cohort + ' | Hemi: ' + hemi)
    plt.barh(y=range(len(sorted_sulci_side_list)), width=[np.mean(dico_stats[sulci]['sizes']) for sulci in sorted_sulci_side_list], height=0.8, align='center', label='average size')
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.xlabel('Number of voxels')
    plt.legend(loc='upper right')
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    if savefig:
        plt.savefig(op.join(path_to_save_figures, cohort_name + '_size_mean.png'))

if 3 in figures:
    plt.figure(3, figsize=(3, 6))
    plt.title('Cohort: ' + cohort + ' | Hemi: ' + hemi)
    err = [np.std(dico_stats[sulci]['sizes']) for sulci in sorted_sulci_side_list]
    plt.plot([0], [0], '+-', color='black', label='standard deviiation')
    plt.barh(y=range(len(sorted_sulci_side_list)), width=[np.mean(dico_stats[sulci]['sizes']) for sulci in sorted_sulci_side_list], height=0.8, align='center', label='average size', xerr=err, capsize=2.0)
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.xlabel('Number of voxels')
    plt.legend(loc='upper right')
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    if savefig:
        plt.savefig(op.join(path_to_save_figures, cohort_name + '_sizes_std.png'))

if 4 in figures:
    plt.figure(4, figsize=(4, 6))
    plt.title('Cohort: ' + cohort + ' | Hemi: ' + hemi)
    x = [dico_stats[sulci]['sizes'] for sulci in sorted_sulci_side_list]
    plt.boxplot(x, vert=False, showfliers=False, notch=False, medianprops={'linewidth': 2},
                boxprops={'color': 'blue'}, capprops={'color': 'blue'}, whiskerprops={'color': 'blue'},
                positions=range(len(sorted_sulci_side_list)))
    plt.yticks(range(len(sorted_sulci_side_list)), [s[:-n] for s in sorted_sulci_side_list], fontsize='x-small')
    plt.xlabel('Number of voxels')
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    if savefig:
        plt.savefig(op.join(path_to_save_figures, cohort_name + '_box_sizes.png'))

plt.show()
