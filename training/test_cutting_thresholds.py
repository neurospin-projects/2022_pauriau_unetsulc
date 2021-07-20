import json
import os
import numpy as np
from sklearn.model_selection import KFold
from training import UnetTrainingSulciLabelling
import time

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

th_range = parameters['th_range']

n_cvinner = parameters['n_cvinner']

model_name = parameters['model_name']

if __name__ == '__main__':

    # # DATA INPUT # #
    print('\nLoading Data\n')

    #Récupération des graphes
    with open(os.path.join(working_path + '/cohorts/cohort-' + cohort_name + '_hemi-' + hemi + '.json'), 'r') as f:
        cohort = json.load(f)
    graphs = []
    notcut_graphs = []
    for s in cohort['subjects']:
        graphs.append(s['graph'])
        notcut_graphs.append(s['notcut_graph'])

    print('graph loaded')

    #Récupération des données (sulci_side_list, dict_sulci, dict_names)
    if os.path.exists(working_path + '/data/' + model_name + '.json'):
        with open(working_path + '/data/' + model_name + '.json', 'r') as f:
            data = json.load(f)
        print('data loaded')
    else:
        data = {}
        print('No Data Found')

    #Récupération du modèle
    print('\nLoading network\n')
    if len(data) == 0:
        method = UnetTrainingSulciLabelling(graphs, hemi, translation_file, cuda=cuda, working_path=working_path, model_name=model_name)
        method.extract_data_from_graphs()
        method.save_data()

    else:
        method = UnetTrainingSulciLabelling(graphs, hemi, translation_file, cuda=cuda, working_path=working_path, model_name=model_name,
                                        dict_names=data['dict_names'], dict_bck2=data['dict_bck2'], sulci_side_list=data['sulci_side_list'])

    model_file = working_path + '/models/' + model_name + '_model.mdsm'
    method.load_saved_model(model_file)


    # # CUTTING # #
    start_time = time.time()

    #Cross Validation pour déterminer la performance du modèle
    kf = KFold(n_splits=n_cvinner, shuffle=True, random_state=0)
    agraphs = np.asarray(graphs)

    notcut_agraphs = np.asarray(notcut_graphs)

    cvi = 1
    for train, test in kf.split(graphs):
        print('\n== Cross Validation {}/{} ==\n'.format(cvi, n_cvinner))
        glist_train = agraphs[train]
        glist_test = agraphs[test]

        print('\nCutting')
        glist_notcut_test = notcut_agraphs[test]
        method.test_thresholds(gfile_list_test=glist_test, gfile_list_notcut_test=glist_notcut_test,
                               threshold_range=th_range)

        cvi += 1

    cv_time = time.time() - start_time
    print('Cross Validation complete in {:.0f}h {:.0f}m {:.0f}s'.format(cv_time // 3600, (cv_time % 3600)//60, (cv_time % 3600)%60))

    path_to_save_results = working_path + '/results/' + model_name + '_cutting_thresholds.json'
    with open(path_to_save_results, 'w') as f:
        json.dump(method.dict_scores, f)
    print('Results saved')

    best_threshold = -1
    best_mean = 0
    for th, scores in method.dict_scores.items():
        mean_score = np.mean(scores)
        print('Threshold', th, sep=' ')
        print('Score', mean_score, sep=' ')
        if mean_score > best_mean:
            best_threshold = th
            best_mean = mean_score
        elif mean_score == best_mean:
            if isinstance(best_threshold, list):
                best_threshold.append(th)
            else:
                best_threshold = [best_threshold, th]

    print('Best threshold:', best_threshold)


