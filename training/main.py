import json
import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from training import UnetTrainingSulciLabelling
import time

# Parameters
with open('parameters.json', 'r') as f:
    parameters = json.load(f)

working_path = parameters['working_path']

cuda = parameters['cuda']

cohort_name = parameters['cohort_name']
hemi = parameters['hemi']
path_to_cohort = parameters['path_to_cohort']
translation_file = parameters['translation_file']

batch_size = parameters['batch_size']

lr = parameters['lr']
momentum = parameters['momentum']
th_range = [0, 1, 2]

n_cvinner = parameters['n_cvinner']
n_epochs = parameters['n_epochs']

if __name__ == '__main__':

    # # DATA INPUT # #
    print('Loading Data\n')

    #Récupération des graphes
    with open(os.path.join(path_to_cohort, 'cohort-' + cohort_name + '_hemi-' + hemi + '.json'), 'r') as f:
        cohort = json.load(f)
    graphs = []
    graphs_notcut = []
    for s in cohort['subjects']:
        graphs.append(s['graph'])
        graphs_notcut.append(s['graph_notcut'])

    #Récupération des données (sulci_side_list, dict_sulci, dict_names)
    if os.path.exists(working_path + '/data.json'):
        with open(working_path + '/data.json', 'r') as f:
            data = json.load(f)
    else:
        data = {}
        print('No Data Found')

    #Récupération du modèle
    if len(data) == 0:
        method = UnetTrainingSulciLabelling(graphs, hemi, translation_file, cuda=cuda, working_path=working_path)
        method.extract_data_from_graphs()
        method.save_data()

    else:
        method = UnetTrainingSulciLabelling(graphs, hemi, translation_file, cuda=cuda, working_path=working_path,
                                        dict_names=data['dict_names'], dict_bck2=data['dict_bck2'], sulci_side_list=data['sulci_side_list'])


    # # TRAINING # #
    print('\n----- Learning -----')
    start_time = time.time()

    #Cross Validation pour déterminer la performance du modèle
    kf = KFold(n_splits=n_cvinner, shuffle=True, random_state=0)
    agraphs = np.asarray(graphs)
    agraphs_notcut = np.asarray(graphs_notcut)
    cvi = 1
    for train, test in kf.split(graphs):
        print('\n== Cross Validation {}/{} =='.format(cvi, n_cvinner))
        glist_train = agraphs[train]
        glist_test = agraphs[test]
        glist_notcut_test = agraphs_notcut[test]

        method.learning(lr=lr, momentum=momentum, num_epochs=n_epochs, gfile_list_train=glist_train,
                    gfile_list_test=glist_test, batch_size=batch_size, save_results=True)

        method.test_thresholds(gfile_list_test=glist_test, gfile_list_notcut_test=glist_notcut_test, threshold_range=th_range)

        cvi += 1

    method.save_results()
    method.save_model()

    print('Cross Validation took %i s.' % (time.time() - start_time))

    with open(working_path+'/results.json', 'r') as f:
        results = json.load(f)

    mean_acc = np.mean(results['best_acc'])
    print('\nFinal Results')
    print('Mean accuracy: ', mean_acc)

    best_threshold = -1
    best_mean = 0

    for th, scores in results['threshold_scores'].items():
        mean_score = np.mean(scores)
        if mean_score > best_mean:
            best_threshold = th
            best_mean = mean_score
        elif mean_score == best_mean:
            best_threshold = [best_threshold, th]

    print(best_threshold)