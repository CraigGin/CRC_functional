"""Helper functions for dataset preparation."""
import os
import json

import pandas as pd
import numpy as np

from scipy.io import mmread
from collections import Counter


def create_taxa_dict(data_dir='CMD_files'):
    """
    Create dictionary of taxonomic abundance tables for each dataset.
    """
    
    file_list = os.listdir(data_dir)
    study_list = list(set([file.split('-')[0] for file in file_list 
                           if file.endswith('feather')]))

    disease_dict = {}

    for study in study_list:
        print('Study: {}'.format(study))

        # Load abundance matrix and meta data
        taxa_abun = pd.read_feather('{}/{}-taxa-abun.feather'.format(data_dir, 
                                                                     study))
        print('Abundance matrix shape: {}'.format(taxa_abun.shape))

        meta_data = pd.read_feather('{}/{}-meta.feather'.format(data_dir,
                                                                study))
        y = (meta_data['study_condition'] == 'CRC').astype(float)
        print('Disease vector shape: {}\n'.format(y.shape))

        # Create dictionary for the study
        study_dict = {'X': taxa_abun,
                      'y': y}

        # Shorten study names
        if study.split('_')[0] == 'ThomasAM':
            study_name = study[:2] + study[-1:]
        else:
            study_name = study[:2]

        # Add study to dictionary if there are both cases and controls
        if y.sum() > 0 and y.sum() < y.shape[0]:
            disease_dict[study_name] = study_dict

    return disease_dict


def create_gf_dict(data_dir='CMD_files'):
    """
    Create dictionary of gene-family abundance tables for each dataset.
    """
    
    file_list = os.listdir(data_dir)
    study_list = list(set([file.split('-')[0] for file in file_list 
                           if file.endswith('feather')]))

    disease_dict = {}

    for study in study_list:
        print('Study: {}'.format(study))

        # Load sparse abundance matrix and names of gene-families
        sp_mat = mmread('{}/{}-gene-families.mtx'.format(data_dir, study))
        with open("{}/{}-gene-family-names.json".format(data_dir, study)) as f:
            gf_names = json.load(f)
        gene_families = pd.DataFrame.sparse.from_spmatrix(sp_mat.T,
                                                          columns=gf_names)
        print('Abundance matrix shape: {}'.format(gene_families.shape))

        # Load meta data and extract study condition
        meta_data = pd.read_feather('{}/{}-meta.feather'.format(data_dir,
                                                                study))
        y = (meta_data['study_condition'] == 'CRC').astype(float)
        print('Disease vector shape: {}\n'.format(y.shape))

        # Create dictionary for the study
        study_dict = {'X': gene_families,
                      'y': y}

        # Shorten study names
        if study.split('_')[0] == 'ThomasAM':
            study_name = study[:2] + study[-1:]
        else:
            study_name = study[:2]

        # Add study to dictionary if there are both cases and controls
        if y.sum() > 0 and y.sum() < y.shape[0]:
            disease_dict[study_name] = study_dict

    return disease_dict


def order_dict(disease_dict):
    """Put CRC studies in the desired order."""
    ordered_studies = ['Thc', 'Ya', 'Yu', 'Tha', 'Thb',
                       'Ze', 'Fe', 'Wi', 'Gu', 'Vo']

    ordered_dict = {}
    for study in ordered_studies:
        ordered_dict[study] = disease_dict[study]
    return ordered_dict


def calculate_stat(X, statistic):
    """Calculate given statistic for a feature table."""
    if statistic == 'max_abundance':
        stat = X.max(axis=0)
    elif statistic == 'prevalence':
        stat = (X > 0).sum(axis=0) / X.shape[0]
    elif statistic == 'variance':
        stat = pd.Series([np.var(X.loc[:, col].to_numpy()) for col in X],
                         index=X.columns)
    else:
        stat = pd.Series(np.ones(X.shape[1]), index=X.columns)
    return stat


def prop(X):
    """Convert sample values to proportions."""
    return X.div(X.sum(axis=1), axis=0)


def filter_by_stat(disease_dict, statistic, threshold, transformation=None):
    """Find features with statistic greater than threshold in more than half the studies."""
    n_studies = np.ceil((len(disease_dict.keys()) + 1) / 2).astype(int)
    print('Finding features with {} greater '.format(statistic) +
          'than {} for at least {} studies'.format(threshold, n_studies))

    above_thr_list = []

    for study in disease_dict:
        X = disease_dict[study]['X']
        if transformation:
            X = transformation(X)
        else:
            print('No transformation')
        stat = calculate_stat(X, statistic)
        above_threshold = stat >= threshold
        above_thr_list.extend(X.columns[above_threshold])

    above_counts = Counter(above_thr_list)
    keep_list = [key for key in above_counts if above_counts[key] >= n_studies]
    if 'UNMAPPED' in keep_list:
        keep_list.remove('UNMAPPED')

    print('{} features meet the criteria\n'.format(len(keep_list)))
    return keep_list


def select_features(disease_dict, feature_list):
    """Select features in feature list."""
    print('Removing Features')
    data_dict = {}
    for study in disease_dict:
        print('Processing study {}'.format(study))
        X = disease_dict[study]['X']
        y = disease_dict[study]['y']

        keep_cols = list(set(feature_list) & set(X.columns))
        X_reduced = X.loc[:, keep_cols]
        if pd.api.types.is_sparse(X_reduced):
            X_reduced = X_reduced.sparse.to_dense()

        print('Shape of new abundance matrix: {}\n'.format(X_reduced.shape))

        data_dict[study] = {'X': X_reduced, 'y': y}
    return data_dict


def group_features(disease_dict, group_dict):
    """Group features with grouping based on group_dict."""
    print('Grouping Features')
    data_dict = {}
    for study in disease_dict:
        print('Processing study {}'.format(study))
        X = disease_dict[study]['X']
        y = disease_dict[study]['y']

        reduced_dict = {}
        for group in group_dict:
            cols = list(set(group_dict[group]) & set(X.columns))
            reduced_dict[group] = X.loc[:, cols].sum(axis=1)

        X_reduced = pd.DataFrame(reduced_dict)
        print('Shape of new abundance matrix: {}\n'.format(X_reduced.shape))

        data_dict[study] = {'X': X_reduced, 'y': y}
    return data_dict


def remove_samples(disease_dict, taxa_dict, min_reads=1e6):
    """Remove samples with read count below min_reads."""
    for study in disease_dict:
        print('Study: {}'.format(study))
        read_cts = taxa_dict[study]['X'].sum(axis=1)
        to_drop = read_cts < min_reads
        print('{} samples dropped due to low read count\n'.format(to_drop.sum()))
        disease_dict[study]['X'] = disease_dict[study]['X'].loc[~to_drop]
        disease_dict[study]['y'] = disease_dict[study]['y'].loc[~to_drop]

    return disease_dict


def combine_abun_tables(disease_dict):
    """Combine abundance tables from each dataset into one."""
    X_list = [disease_dict[study]['X'] for study in disease_dict]
    X_all = pd.concat(X_list,
                      axis=0,
                      join='outer',
                      ignore_index=False).fillna(0)

    studies = [[study] * disease_dict[study]['X'].shape[0]
               for study in disease_dict]
    studies = sum(studies, [])
    return X_all, studies


def add_zero_columns(disease_dict):
    """Add columns of zeros for those not present in each dataset."""
    X_all, studies = combine_abun_tables(disease_dict)

    for study in disease_dict:
        print('Study: {}'.format(study))
        X = X_all.iloc[[i for i in range(len(studies))
                        if studies[i] == study], :]
        y = disease_dict[study]['y']

        study_dict = {'X': X,
                      'y': y}

        disease_dict[study] = study_dict
        print('Abundance matrix shape: {}'.format(X.shape))

    return disease_dict


def get_all_features(disease_dict):
    """Get list of the union of features in all datasets."""
    feature_list = []

    for study in disease_dict:
        study_features = list(disease_dict[study]['X'].columns)
        feature_list = list(set(feature_list + study_features))

    return feature_list
