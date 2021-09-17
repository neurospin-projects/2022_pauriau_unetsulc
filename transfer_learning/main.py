import json
import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from transfer_learning import UnetTransferSulciLabelling
import time
import warnings

warnings.filterwarnings(action='ignore', message='the number of', category=UserWarning)

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

dict_model = parameters['dict_model']
model_name = dict_model['name']
print('Model name: ', model_name)

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
                print('no not cut graphs')
    print(len(graphs), 'graph loaded', sep=' ')

    #Récupération des données (sulci_side_list, dict_sulci, dict_names)
    path_to_data = working_path + '/data/' + cohort_file + '.json'
    if os.path.exists(path_to_data):
        with open(path_to_data, 'r') as f:
            data = json.load(f)
        print('data loaded')
    else:
        data = {}
        print('no data found')

    #Récupération du modèle
    print('\nLoading network\n')
    if len(data) == 0:
        method = UnetTransferSulciLabelling(graphs, hemi, translation_file, cuda=cuda, working_path=working_path, dict_model=dict_model)
        method.extract_data_from_graphs()
        method.save_data(name=cohort_file)

    else:
        method = UnetTransferSulciLabelling(graphs, hemi, translation_file, cuda=cuda, working_path=working_path, dict_model=dict_model,
                                            dict_names=data['dict_names'], dict_bck2=data['dict_bck2'], sulci_side_list=data['sulci_side_list'])

    # # TRAINING # #
    print('\n----- Learning -----')
    start_time = time.time()

    #Cross Validation pour déterminer la performance du modèle
    kf = KFold(n_splits=n_cvinner, shuffle=True, random_state=0)
    agraphs = np.asarray(graphs)

    if notcut_graphs is not None:
        notcut_agraphs = np.asarray(notcut_graphs)

    for cvi, (train, test) in enumerate(kf.split(graphs)):
        print('\n== Cross Validation {}/{} ==\n'.format(cvi, n_cvinner-1))
        glist_train = agraphs[train]
        glist_test = agraphs[test]

        method.learning(lr=lr, momentum=momentum, num_epochs=n_epochs, gfile_list_train=glist_train,
                    gfile_list_test=glist_test, batch_size=batch_size, patience=patience, save_results=True)

        if notcut_graphs is not None:
            print('\nCutting')
            glist_notcut_test = notcut_agraphs[test]
            method.test_thresholds(gfile_list_test=glist_test, gfile_list_notcut_test=glist_notcut_test,
                                   threshold_range=th_range)

        method.save_model(name=model_name+'_cv'+str(cvi))

    method.save_results()
    cv_time = time.time() - start_time
    print('Cross Validation complete in {:.0f}h {:.0f}m {:.0f}s'.format(cv_time // 3600, (cv_time % 3600)//60, (cv_time % 3600)%60))

    with open(working_path+'/results/' + model_name + '.json', 'r') as f:
        results = json.load(f)

    mean_acc = np.mean(results['best_acc'])
    print('\nFinal Results')
    print('Mean accuracy: ', mean_acc)

    if notcut_graphs is not None:
        best_thresholds, best_means = [], []
        for th, scores in results['threshold_scores'].items():
            mean_scores = np.mean(scores, axis=1)
            for n, sc in enumerate(mean_scores):
                if len(best_means) < n + 1:
                    best_means.append(sc)
                    best_thresholds.append(th)
                elif sc > best_means[n]:
                    best_thresholds[n] = th
                    best_means[n] = sc
                elif sc == best_means[n]:
                    if isinstance(best_thresholds[n], list):
                        best_thresholds[n].append(th)
                    else:
                        best_thresholds[n] = [best_thresholds[n], th]
        for n, th in enumerate(best_thresholds):
            print('Training n°', n, ' | Best threshold:', th)
            if isinstance(th, list):
                th = np.random.choice(th)
            method.save_params(best_threshold=th, name=model_name+'_cv'+str(n))
        for th in best_thresholds:
            if isinstance(th, list):
                best_thresholds += th
                best_thresholds.remove(th)
        best_th = max(set(best_thresholds), key=best_thresholds.count)
        method.save_params(best_th)
        print('\nBest Threshold: ', best_th)
