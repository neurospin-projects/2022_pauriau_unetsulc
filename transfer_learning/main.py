import json
import os
from sklearn.model_selection import KFold, train_test_split
from transfer_learning import UnetTransferSulciLabelling

# DATA INPUT
working_path = '/volatile/home/pierre/Programmes/transfer_learning'

cohort_name = 'Chimp_3T'
hemi = 'L'
path_to_cohort = '/volatile/home/pierre/Data/cohorts'
translation_file = '/casa/host/build/share/brainvisa-share-5.1/nomenclature/translation/sulci_model_2018.trl'

with open(os.path.join(path_to_cohort, 'cohort-' + cohort_name + '_hemi-' + hemi + '.json'), 'r') as f:
    cohort = json.load(f)

graphs = []
for s in cohort['subjects']:
    graphs.append(s['graph'])
# graphs = graphs[:10]
# with open(working_path + '/data.json', 'r') as f:
#     data = json.load(f)


if __name__ == '__main__':

    method = UnetTransferSulciLabelling(graphs, hemi, translation_file, working_path=working_path)

    # method = UnetTransferSulciLabelling(graphs, hemi, translation_file, working_path=working_path,
    #                                     dict_names=data['dict_names'], dict_bck2=data['dict_bck2'], sulci_side_list=data['sulci_side_list'])

    method.extract_data_from_graphs()

    # # FIND HYPERPARAMETER # #
    # lr_range = [1e-5, 1e-2]
    # momentum_range = [0.6, 0.8]


    method.load_model()

    gfile_list_train, gfile_list_test = train_test_split(graphs, test_size=0.1)
    method.learning(lr=1e-3, momentum=0.9, num_epochs=5, gfile_list_train=gfile_list_train, gfile_list_test=gfile_list_test, batch_size=1, save_results=True)
    method.save_data()
    method.save_results()
    method.save_model()


    # n_cvinner = 3
    # kf = KFold(n_splits=n_cvinner, shuffle=True, random_state=0)
    # for train, test in kf.split(graphs):