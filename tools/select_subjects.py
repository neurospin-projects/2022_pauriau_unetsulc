# -*- coding: utf-8 -*-

import os
import os.path as op
import json
import random

# ----------------------------------------------------------------------- #
# Functions to create cohorts with some subjects which have not all files
# ----------------------------------------------------------------------- #

def select_subjects(env_file, name_cohort, save):
    """ Create the exclusion list of a cohort, regardless if the subject have the good files or not
    """
    dico = {}
    dico['exclusion_list'] = []
    dico['inclusion_list'] = []
    sbj = 0

    with open(env_file, 'r') as f:
        param = json.load(f)
    db_dir = param["cohorts"][name_cohort]['path']
    acquisition = param["cohorts"][name_cohort]['acquisition']
    center = param["cohorts"][name_cohort]['centers']
    analysis = param["cohorts"][name_cohort]['analysis']
    graph_v = param["cohorts"][name_cohort]['graph_v']
    ngraph_v = param["cohorts"][name_cohort]['ngraph_v']
    session = param["cohorts"][name_cohort]['session']

    path = os.path.join(db_dir, center)

    print('Subjets excluded: ')
    for s in os.listdir(path):
        if s[-4:] != "minf" and s[-4:] != "html":
            sbj += 1
            hemi = 'L'

            # T1
            if not (op.exists(op.join(db_dir, center, s, 't1mri', acquisition, s + ".nii"))
                      or op.exists(op.join(db_dir, center, s, 't1mri', acquisition, s + ".nii.gz"))):
                dico['exclusion_list'].append(s)
                print(s, ': No T1')
            # Roots
            elif not (op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation',
                                        hemi + 'roots_' + s + '.nii'))
                      or op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation',
                                           hemi + 'roots_' + s + '.nii.gz'))):
                dico['exclusion_list'].append(s)
                print(s, ': No roots')
            # Skeleton
            elif not (op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'skeleton_' + s + '.nii'))
                      or op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'skeleton_' + s + '.nii.gz'))):
                dico['exclusion_list'].append(s)
                print(s, ': No skeleton')
            # Graph
            elif not op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'folds', graph_v, session, hemi + s + '_' + session + '.arg')):
                dico['exclusion_list'].append(s)
                print(s, ': No graph')

            # Not cut graph
            elif not op.exists(
                    op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'folds', ngraph_v, hemi + s + '.arg')):
                if ngraph_v != -1:
                    dico['exclusion_list'].append(s)
                    print('No not cut graph')
            else:
                dico['inclusion_list'].append(s)

    print('Nombre total de sujets: ', sbj)
    print('Nombre de sujets exclus: ', len(dico['exclusion_list']))
    print('Nombre de sujets inclus:', len(dico['inclusion_list']))

    if save:
        with open(env_file, 'r') as f:
            env = json.load(f)
            env["cohorts"][name_cohort]["exclusion"] = dico['exclusion_list']
        with open(env_file, 'w') as f:
            json.dump(env, f)
        print('Saved in ', env_file)


def change_path_cohort(cohort_file, new_path, start, end, path_to_save, save):
    """ Modify the path of attributes of all subjects in the cohort
    """
    with open(cohort_file, 'r') as f:
        cohort = json.load(f)
    for sbj in cohort['subjects']:
        sbj['t1'] = sbj['t1'][:start] + new_path + sbj['t1'][end:]
        sbj['roots'] = sbj['roots'][:start] + new_path + sbj['roots'][end:]
        sbj['skeleton'] = sbj['skeleton'][:start] + new_path + sbj['skeleton'][end:]
        sbj['graph'] = sbj['graph'][:start] + new_path + sbj['graph'][end:]
        if sbj['notcut_graph'] != -1:
            sbj['notcut_graph'] = sbj['notcut_graph'][:start] + new_path + sbj['notcut_graph'][end:]

    print(cohort['subjects'][0]['t1'])
    print(cohort['subjects'][0]['roots'])
    print(cohort['subjects'][0]['skeleton'])
    print(cohort['subjects'][0]['graph'])
    print(cohort['subjects'][0]['notcut_graph'])
    if save:
        with open(path_to_save, 'w') as f:
            json.dump(cohort, f)
        print('cohort save')


def create_short_cohort(path_to_cohort, n_sbj, order, save, new_name=None):
    """ Create cohort with n_sbj subjects from a full cohort
    """
    with open(path_to_cohort, 'r') as f:
        cohort = json.load(f)
    if order:
        cohort['subjects'] = cohort['subjects'][:n_sbj]
    else:
        cohort['subjects'] = random.sample(cohort['subjects'], k=n_sbj)
    if new_name is None:
        new_cohort['name'] = cohort['name'][:-7] + '_short' + cohort['name'][-7:]
        new_path = path_to_cohort[:-12] + '_short' + path_to_cohort[-12:]
    else:
        new_cohort['name'] = cohort['name'][:-7] + new_name + cohort['name'][-7:]
        new_path = path_to_cohort[:-12] + new_name + path_to_cohort[-12:]
    if save:
        with open(new_path, 'w') as f:
            json.dump(new_cohort, f)
        print('cohort saved:', new_path)
    return new_cohort


def select_subjects_from_cohort(path_to_cohort, sbj, save, new_name=None):
    """ Create cohort with subjects in sbj from the cohort
    """
    with open(path_to_cohort, 'r') as f:
        cohort = json.load(f)
    new_cohort = cohort.copy()
    new_cohort['subjects'] = []
    for s in cohort['subjects']:
        if s['name'] in sbj:
            new_cohort['subjects'].append(s)
        else:
            print(s['name'])
    if new_name is None:
        new_cohort['name'] = cohort['name'][:-7] + '_short' + cohort['name'][-7:]
        new_path = path_to_cohort[:-12] + '_short' + path_to_cohort[-12:]
    else:
        new_cohort['name'] = cohort['name'][:-7] + new_name + cohort['name'][-7:]
        new_path = path_to_cohort[:-12] + new_name + path_to_cohort[-12:]
    if save:
        with open(new_path, 'w') as f:
            json.dump(new_cohort, f)
        print('cohort saved:', new_path)
    return new_cohort


def create_cohort(env_file, name_cohort):
    """ Create cohort called named_cohort from env_file, do not take in account subjects without the good files
    """

    with open(env_file, 'r') as f:
        param = json.load(f)
    db_dir = param["cohorts"][name_cohort]['path']
    acquisition = param["cohorts"][name_cohort]['acquisition']
    center = param["cohorts"][name_cohort]['centers']
    analysis = param["cohorts"][name_cohort]['analysis']
    graph_v = pasbj is not Noneram["cohorts"][name_cohort]['graph_v']
    ngraph_v = param["cohorts"][name_cohort]['ngraph_v']
    session = param["cohorts"][name_cohort]['session']
    working_path = param["working_path"]

    path = os.path.join(db_dir, center)

    for hemi in ['L', 'R']:
        cohort = {'name': name_cohort+'_hemi-'+hemi, 'subjects': []}
        for s in os.listdir(path):
            if s[-4:] != "minf" and s[-4:] != "html":
                to_add = True
                name = s
                # T1
                if op.exists(op.join(db_dir, center, s, 't1mri', acquisition, s + ".nii")):
                    t1 = op.join(db_dir, center, s, 't1mri', acquisition, s + ".nii")
                elif op.exists(op.join(db_dir, center, s, 't1mri', acquisition, s + ".nii.gz")):
                    t1 = op.join(db_dir, center, s, 't1mri', acquisition, s + ".nii.gz")
                else:
                    to_add = False
                    print(name, 'No T1')

                # Roots
                if op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'roots_' + s + '.nii')):
                    roots = op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'roots_' + s + '.nii')
                elif op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'roots_' + s + '.nii.gz')):
                    roots = op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'roots_' + s + '.nii.gz')
                else:
                    to_add = False
                    print(name, 'No roots')

                # Skeleton
                if op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'skeleton_' + s + '.nii')):
                    skeleton = op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'skeleton_' + s + '.nii')
                elif op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'skeleton_' + s + '.nii.gz')):
                    skeleton = op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'segmentation', hemi + 'skeleton_' + s + '.nii.gz')
                else:
                    to_add = False
                    print(name, 'No skeleton')

                # Graph
                if op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'folds', graph_v, session, hemi + s + '_' + session + '.arg')):
                    graph = op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'folds', graph_v, session, hemi + s + '_' + session + '.arg')
                else:
                    to_add = False
                    print(name, 'No graph')

                # Not cut graph
                if op.exists(op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'folds', ngraph_v, hemi + s + '.arg')):
                    notcut_graph = op.join(db_dir, center, s, 't1mri', acquisition, analysis, 'folds', ngraph_v, hemi + s + '.arg')
                else:
                    if ngraph_v != -1:
                        to_add = False
                        print(name, 'No not cut graph')
                    else:
                        notcut_graph = None

                if to_add:
                    dico_sbj = {'name': name,
                                 't1': t1,
                                 'roots': roots,
                                 'skeleton': skeleton,
                                 'graph': graph,
                                 'notcut_graph': notcut_graph}
                    cohort['subjects'].append(dico_sbj)
                    print('subject', name, 'added')

        print('Cohort: ', name_cohort)
        print('Hemi: ', hemi)
        print('Number of subject: ', len(cohort['subjects']), '\n')

        with open(op.join(working_path, 'cohorts', 'cohort-'+name_cohort+'_hemi-'+hemi+'.json'), 'w') as f:
            json.dump(cohort, f)
        print('File saved :' + 'cohort-'+name_cohort+'_hemi-'+hemi+'.json\n')


def create_composed_cohort(env_file, name_cohort):
    """ Create cohort called named cohort from different cohorts, sum up in the env_file
    """
    with open(env_file, 'r') as f:
        param = json.load(f)
    working_path = param["working_path"]
    for hemi in ['L', 'R']:
        cohort = {'name': name_cohort + '_hemi-' + hemi, 'subjects': []}
        for n, v in param["composed_cohorts"][name_cohort]["cohort"].items():
            with open(op.join(working_path, 'cohorts', 'cohort-'+n+'_hemi-'+hemi+'.json'), 'r') as f:
                c = json.load(f)
            subjects = c['subjects']
            if 'indexes' in v.keys():
                for i in v['indexes']:
                    cohort['subjects'].append(subjects[i])
            else:
                cohort['subjects'] += subjects

        print('\nCohort: ', name_cohort)
        print('Hemi: ', hemi)
        print('Number of subject: ', len(cohort['subjects']))

        with open(op.join(working_path, 'cohorts', 'cohort-' + name_cohort + '_hemi-' + hemi + '.json'), 'w') as f:
            json.dump(cohort, f)
        print('File saved : ', op.join(working_path, 'cohorts', 'cohort-' + name_cohort + '_hemi-' + hemi + '.json'))
