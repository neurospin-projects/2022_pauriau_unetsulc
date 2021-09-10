import json
import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from training import UnetTrainingSulciLabelling
import time
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
# Parameters
with open('parameters.json', 'r') as f:
    parameters = json.load(f)

working_path = parameters['working_path']
print('working directory: ', working_path)

cuda = parameters['cuda']

cohort_name = parameters['cohort_name']
print('cohort: ', cohort_name)
hemi = parameters['hemi']
print('hemi: ', hemi)
translation_file = parameters['translation_file']

batch_size = parameters['batch_size']

lr = parameters['lr']
momentum = parameters['momentum']
th_range = parameters['th_range']

n_cvinner = parameters['n_cvinner']
n_epochs = parameters['n_epochs']

model_name = parameters['model_name']
print('model: ', model_name)

if 'patience' in parameters.keys():
    patience = parameters['patience']
else:
    patience = None

if __name__ == '__main__':

    # # DATA INPUT # #
    print('\nLoading Data\n')

    #Récupération des graphes
    cohort_file = 'cohort-' + cohort_name + '_hemi-' + hemi
    with open(os.path.join(working_path + '/cohorts/' + cohort_file + '.json'), 'r') as f:
        cohort = json.load(f)
    graphs = []
    notcut_graphs = []
    for s in cohort['subjects']:
        graphs.append(s['graph'])
        if notcut_graphs is not None :
            if s['notcut_graph'] is not None:
                notcut_graphs.append(s['notcut_graph'])
            else:
                notcut_graphs = None
                print('No not cut graphs')
    print('graph loaded')

    #Récupération des données (sulci_side_list, dict_sulci, dict_names)
    path_to_data = working_path + '/data/' + cohort_file +'.json'
    if os.path.exists(path_to_data):
        with open(path_to_data, 'r') as f:
            data = json.load(f)
        print('Data Loaded')
    else:
        data = {}
        print('No Data Found')

    #Récupération du modèle
    print('\nLoading network\n')
    if len(data) == 0:
        method = UnetTrainingSulciLabelling(graphs, hemi, translation_file, cuda=cuda, working_path=working_path, model_name=model_name)
        method.extract_data_from_graphs()
        method.save_data(name=cohort_file)

    else:
        method = UnetTrainingSulciLabelling(graphs, hemi, translation_file, cuda=cuda, working_path=working_path, model_name=model_name,
                                        dict_names=data['dict_names'], dict_bck2=data['dict_bck2'], sulci_side_list=data['sulci_side_list'])


    # # TRAINING # #
    print('\n----- Learning -----')
    start_time = time.time()

    #Cross Validation pour déterminer la performance du modèle
    kf = KFold(n_splits=n_cvinner, shuffle=True, random_state=0)
    agraphs = np.asarray(graphs)

    if notcut_graphs is not None:
        notcut_agraphs = np.asarray(notcut_graphs)

    cvi = 1
    for train, test in kf.split(graphs):
        print('\n== Cross Validation {}/{} ==\n'.format(cvi, n_cvinner))
        glist_train = agraphs[train]
        glist_test = agraphs[test]

        method.learning(lr=lr, momentum=momentum, num_epochs=n_epochs, gfile_list_train=glist_train,
                    gfile_list_test=glist_test, batch_size=batch_size, patience=patience, save_results=True)

        if notcut_graphs is not None:
            print('\nCutting')
            glist_notcut_test = notcut_agraphs[test]
            method.test_thresholds(gfile_list_test=glist_test, gfile_list_notcut_test=glist_notcut_test,
                                   threshold_range=th_range)

        method.save_model(name=model_name+'_'+'cv'+str(cvi))

        cvi += 1

    method.save_results()
    cv_time = time.time() - start_time

    print('Cross Validation complete in {:.0f}h {:.0f}m {:.0f}s'.format(cv_time // 3600, (cv_time % 3600)//60, (cv_time % 3600)%60))

    with open(working_path+'/results/' + model_name + '.json', 'r') as f:
        results = json.load(f)

    mean_acc = np.mean(results['best_acc'])
    print('\nFinal Results')
    print('Mean accuracy: ', mean_acc)

    if notcut_graphs is not None:
        best_threshold = -1
        best_mean = 0
        for th, scores in results['threshold_scores'].items():
            for n, sc in enumerate(scores):
                mean_score = np.mean(sc)
                if mean_score > best_mean:
                    best_threshold = th
                    best_mean = mean_score
                elif mean_score == best_mean:
                    if isinstance(best_threshold, list):
                        best_threshold.append(th)
                    else:
                        best_threshold = [best_threshold, th]
                print('Training n°', n, ' | Best threshold:', best_threshold)