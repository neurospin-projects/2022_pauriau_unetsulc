import json
import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from transfer_learning import UnetTransferSulciLabelling
import time

# Parameters
with open('parameters.json', 'r') as f:
    parameters = json.load(f)

working_path = parameters['working_path']

cohort_name = parameters['cohort_name']
hemi = parameters['hemi']
path_to_cohort = parameters['path_to_cohort']
translation_file = parameters['translation_file']

batch_size = parameters['batch_size']

lr = parameters['lr']
momentum = parameters['momentum']

n_cvinner = parameters['n_cvinner']
n_epochs = parameters['n_epochs']

if __name__ == '__main__':

    # # DATA INPUT # #
    print('Loading Data\n')

    #Récupération des graphes
    with open(os.path.join(path_to_cohort, 'cohort-' + cohort_name + '_hemi-' + hemi + '.json'), 'r') as f:
        cohort = json.load(f)
    graphs = []
    for s in cohort['subjects']:
        graphs.append(s['graph'])

    #Récupération des données (sulci_side_list, dict_sulci, dict_names)
    if os.path.exists(working_path + '/data.json'):
        with open(working_path + '/data.json', 'r') as f:
            data = json.load(f)
    else:
        data = {}
        print('No Data Found')

    #Récupération du modèle
    if len(data) == 0:
        method = UnetTransferSulciLabelling(graphs, hemi, translation_file, working_path=working_path)
        method.extract_data_from_graphs()
        method.save_data()

    else:
        method = UnetTransferSulciLabelling(graphs, hemi, translation_file, working_path=working_path,
                                        dict_names=data['dict_names'], dict_bck2=data['dict_bck2'], sulci_side_list=data['sulci_side_list'])


    # # TRAINING # #
    print('\nFinal Training')
    start_time = time.time()
    #Cross Validation pour déterminer la performance du modèle
    kf = KFold(n_splits=n_cvinner, shuffle=True, random_state=0)
    agraphs = np.asarray(graphs)
    cvi = 1
    for train, test in kf.split(graphs):
        print('\n== Cross Validation {}/{} =='.format(cvi, n_cvinner))
        glist_train = agraphs[train]
        glist_test = agraphs[test]
        method.learning(lr=lr, momentum=momentum, num_epochs=n_epochs, gfile_list_train=glist_train,
                    gfile_list_test=glist_test, batch_size=batch_size, save_results=True)
        cvi += 1

    method.save_results()
    method.save_model()

    print('Cross Validation took %i s.' % (time.time() - start_time))

    with open(working_path+'/results.json', 'r') as f:
        results = json.load(f)

    mean_acc = np.mean(results['best_acc'])
    print('\nFinal Results')
    print('Mean accuracy: ', mean_acc)