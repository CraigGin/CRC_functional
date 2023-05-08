"""Functions used for training/evaluating models."""
import pickle
import warnings
import os
import shap

import pandas as pd
import numpy as np

from random import sample
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from .transformations import data_transformation


def load_data(data_name):
    """Load data."""
    with open('data/{}.pkl'.format(data_name), 'rb') as f:
        data_dict = pickle.load(f)

    return data_dict


def split_data(data_dict, test_dataset, balanced=True, validation=False):
    """Split data into train/test (and optionally validation)."""

    # Balance the amount of data from each dataset
    if balanced:
        size_dict = {dataset: data_dict[dataset]['y'].shape[0]
                     for dataset in data_dict
                     if dataset != test_dataset}
        sample_size = min(size_dict.values())
        index_dict = {dataset: sample(range(data_dict[dataset]['y'].shape[0]),
                                      sample_size)
                      for dataset in size_dict}
        X_list = [data_dict[dataset]['X'].iloc[index_dict[dataset], :]
                  for dataset in index_dict]
        y_list = [data_dict[dataset]['y'].iloc[index_dict[dataset]]
                  for dataset in index_dict]
        X_train_val = pd.concat(X_list)
        y_train_val = pd.concat(y_list)
    # Use all data from all datasets
    else:
        X_train_val = pd.concat([data_dict[dataset]['X']
                                 for dataset in data_dict
                                 if dataset != test_dataset])
        y_train_val = pd.concat([data_dict[dataset]['y']
                                 for dataset in data_dict
                                 if dataset != test_dataset])

    X_test = data_dict[test_dataset]['X']
    y_test = data_dict[test_dataset]['y']

    # Further split training into train/validation
    if validation:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                          y_train_val,
                                                          test_size=0.2)
        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train_val, y_train_val, X_test, y_test


def feature_importance(model, X_val, y_val, importance_measure):
    """Calculate feature importance."""
    if importance_measure == 'SHAP':
        explainer = shap.TreeExplainer(model)
        importance_dict = {'shap_values': explainer.shap_values(X_val,
                                                                check_additivity=False),
                           'expected_value': explainer.expected_value}
    elif importance_measure == 'permutation':
        perm_importance = permutation_importance(model, X_val, y_val, n_repeats=10)
        importance_dict = {'permutation': perm_importance['importances_mean']}
    elif importance_measure is None:
        importance_dict = {}
    else:
        importance_dict = {}
        warnings.warn('Invalid importance measure.')
    return importance_dict
    

def intra_dataset(data_name='CRC_taxa',
                  file_suffix='',
                  num_repetitions=50,
                  num_folds=5,
                  transformation_type='prop',
                  transformation_opts=dict(),
                  model_opts={'n_estimators': 500, 'max_features': 'sqrt'}):
    """Perform intra-dataset prediction for all datasets."""
    save_dir = 'results/{}/intra/'.format(data_name)

    # If the directory does not exist, create it
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data_dict = load_data(data_name)

    # Perform intra-dataset evaluation
    results_dict = {dataset: [] for dataset in data_dict}
    for dataset in data_dict:
        print('Training on dataset {}'.format(dataset))
        X_train_val = data_dict[dataset]['X']
        y_train_val = data_dict[dataset]['y']

        # Repeated stratifies K-fold cross-validation
        kf = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_repetitions)

        for i, (train_index, val_index) in enumerate(kf.split(X_train_val, y_train_val)):
            X_train, X_val = (X_train_val.iloc[train_index],
                              X_train_val.iloc[val_index])
            y_train, y_val = (y_train_val.iloc[train_index],
                              y_train_val.iloc[val_index])

            # Skip if only one label in training data
            if len(set(y_train)) < 2:
                warnings.warn('Skipping fold - only one label')
                continue

            # Transform training data
            X_train, test_opts = data_transformation(X_train,
                                                     transformation_type,
                                                     transformation_opts)

            # Train model
            model = RandomForestClassifier(**model_opts)
            model.fit(X_train.to_numpy(), y_train)

            # Transform validation data
            X_val, _ = data_transformation(X_val,
                                           transformation_type,
                                           test_opts)

            # Calculate AUC if there are both labels in validation
            if len(set(y_val)) == 2:
                y_pred = model.predict_proba(X_val.to_numpy())[:, 1]
                AUC = roc_auc_score(y_val, y_pred)
            else:
                AUC = np.nan
                warnings.warn('Only one class in test set. '
                              + 'AUC not calculated.')

            # Calculate validation accuracy
            val_accuracy = model.score(X_val.to_numpy(), y_val)

            # Add to dictionary
            iteration_dict = {'AUC': AUC,
                              'accuracy': val_accuracy}

            results_dict[dataset].append(iteration_dict)

    # Save results and params dictionaries
    results_file = '{}results{}.pkl'.format(save_dir, file_suffix)
    with open(results_file, 'wb') as fp:
        pickle.dump(results_dict, fp)

    params_dict = {'data_name': data_name,
                   'num_repetitions': num_repetitions,
                   'num_folds': num_folds,
                   'transformation_type': transformation_type,
                   'transformation_opts': transformation_opts,
                   'model_opts': model_opts}

    params_file = '{}params{}.pkl'.format(save_dir, file_suffix)
    with open(params_file, 'wb') as fp:
        pickle.dump(params_dict, fp)


def cross_dataset(data_name='CRC_taxa',
                  file_suffix='',
                  num_repetitions=50,
                  transformation_type='prop',
                  transformation_opts=dict(),
                  model_opts={'n_estimators': 500, 'max_features': 'sqrt'},
                  importance_measure='SHAP'):
    """Perform cross-dataset prediction for all studies."""
    save_dir = 'results/{}/cross/'.format(data_name)

    # If the directory does not exist, create it
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data_dict = load_data(data_name)

    # Perform cross-dataset evaluation
    results_dict = {}
    importance_dict = {}
    for dataset in data_dict:
        print('Training on dataset {}'.format(dataset))
        X_train = data_dict[dataset]['X']
        y_train = data_dict[dataset]['y']

        # Skip if only one label in training data
        if len(set(y_train)) < 2:
            warnings.warn('Skipping dataset - only one label')
            continue

        # Transform training data
        X_train, test_opts = data_transformation(X_train,
                                                 transformation_type,
                                                 transformation_opts)

        dataset_dict = {test_dataset: [] for test_dataset in data_dict}
        dataset_imp_dict = {}
        # Train model and evaluate AUC num_repetitions times
        for i in range(num_repetitions):
            # Train model
            model = RandomForestClassifier(**model_opts)
            model.fit(X_train.to_numpy(), y_train)

            # Calculate AUC and accuracy for each test dataset
            for test_dataset in data_dict:
                if test_dataset == dataset:
                    AUC = 0
                    test_accuracy = 0
                else:
                    X_test = data_dict[test_dataset]['X']
                    y_test = data_dict[test_dataset]['y']

                    # Transform test data
                    X_test, _ = data_transformation(X_test,
                                                    transformation_type,
                                                    test_opts)

                    # Calculate AUC if there are both labels in validation
                    if len(set(y_test)) == 2:
                        y_pred = model.predict_proba(X_test.to_numpy())[:, 1]
                        AUC = roc_auc_score(y_test, y_pred)
                    else:
                        AUC = np.nan
                        warnings.warn('Only one class in test set. '
                                      + 'AUC not calculated.')

                    # Calculate test accuracy
                    test_accuracy = model.score(X_test.to_numpy(), y_test)

                # Add results to dictionary
                dataset_dict[test_dataset].append({'AUC': AUC,
                                                   'accuracy': test_accuracy})

            # Calculate feature importance for the first iteration
            if i == 0:
                for test_dataset in data_dict:
                    if test_dataset == dataset:
                        f_imp = feature_importance(model,
                                                   X_train,
                                                   y_train,
                                                   importance_measure)
                    else:
                        X_test = data_dict[test_dataset]['X']
                        y_test = data_dict[test_dataset]['y']

                        X_test, _ = data_transformation(X_test,
                                                        transformation_type,
                                                        test_opts)

                        f_imp = feature_importance(model,
                                                   X_test,
                                                   y_test,
                                                   importance_measure)

                    dataset_imp_dict[test_dataset] = {'importance': f_imp}

        results_dict[dataset] = dataset_dict
        importance_dict[dataset] = dataset_imp_dict

    # Save results and params dictionaries
    results_file = '{}results{}.pkl'.format(save_dir, file_suffix)
    with open(results_file, 'wb') as fp:
        pickle.dump(results_dict, fp)

    importance_file = '{}importance{}.pkl'.format(save_dir, file_suffix)
    with open(importance_file, 'wb') as fp:
        pickle.dump(importance_dict, fp)

    params_dict = {'data_name': data_name,
                   'num_repetitions': num_repetitions,
                   'transformation_type': transformation_type,
                   'transformation_opts': transformation_opts,
                   'model_opts': model_opts,
                   'importance_measure': importance_measure}

    params_file = '{}params{}.pkl'.format(save_dir, file_suffix)
    with open(params_file, 'wb') as fp:
        pickle.dump(params_dict, fp)


def LODO(data_name='CRC_taxa',
         file_suffix='',
         num_repetitions=50,
         transformation_type='prop',
         transformation_opts=dict(),
         model_opts={'n_estimators': 500, 'max_features': 'sqrt'},
         balanced=True):
    """Perform leave-one-dataset-out prediction."""
    save_dir = 'results/{}/LODO/'.format(data_name)

    # If the directory does not exist, create it
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    data_dict = load_data(data_name)

    # Perform LODO evaluation
    results_dict = {}
    for test_dataset in data_dict:
        print('Testing on dataset {}'.format(test_dataset))

        results_dict[test_dataset] = []
        for i in range(num_repetitions):
            # Split data into train and test
            X_train, y_train, X_test, y_test = split_data(data_dict,
                                                          test_dataset,
                                                          balanced,
                                                          validation=False)

            # Skip if only one label in training data
            if len(set(y_train)) < 2:
                warnings.warn('Skipping repetition - only one label')
                continue

            # Transform training data
            X_train, test_opts = data_transformation(X_train,
                                                     transformation_type,
                                                     transformation_opts)

            # Transform test data
            X_test, _ = data_transformation(X_test,
                                            transformation_type,
                                            test_opts)

            # Train model
            model = RandomForestClassifier(**model_opts)
            model.fit(X_train.to_numpy(), y_train)

            # Calculate AUC if there are both labels in validation
            if len(set(y_test)) == 2:
                y_pred = model.predict_proba(X_test.to_numpy())[:, 1]
                AUC = roc_auc_score(y_test, y_pred)
            else:
                AUC = np.nan
                warnings.warn('Only one class in test set. '
                              + 'AUC not calculated.')

            # Calculate test accuracy
            accuracy = model.score(X_test.to_numpy(), y_test)

            # Add results to dictionary
            results_dict[test_dataset].append({'AUC': AUC,
                                               'accuracy': accuracy})

    # Save results and params dictionaries
    results_file = '{}results{}.pkl'.format(save_dir, file_suffix)
    with open(results_file, 'wb') as fp:
        pickle.dump(results_dict, fp)

    params_dict = {'data_name': data_name,
                   'num_repetitions': num_repetitions,
                   'transformation_type': transformation_type,
                   'transformation_opts': transformation_opts,
                   'model_opts': model_opts,
                   'balanced': balanced}

    params_file = '{}params{}.pkl'.format(save_dir, file_suffix)
    with open(params_file, 'wb') as fp:
        pickle.dump(params_dict, fp)
