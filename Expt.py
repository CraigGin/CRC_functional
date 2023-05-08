"""Train Microbiome classifiers with Random Forests."""
import argparse
from utils.training import intra_dataset, cross_dataset, LODO


# From https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            if value.isdigit():
                value = int(value)
            elif value == 'None':
                value = None
            else:
                try:
                    value = float(value)
                except:
                    pass
            getattr(namespace, self.dest)[key] = value


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--analysis_type',
                    nargs='+',
                    default='all',
                    help='intra, cross, lodo, or all')
parser.add_argument('-dn', '--data_name',
                    help='name of dataset, should be saved as data/data_name.pkl')
parser.add_argument('--file_suffix',
                    default='',
                    help='suffix for results file')
parser.add_argument('-nr', '--num_repetitions',
                    type=int,
                    default=50,
                    help='number of times to repeat experiment')
parser.add_argument('-nf', '--num_folds',
                    type=int,
                    default=5,
                    help='number of folds for k-fold cross-validation')
parser.add_argument('-tt', '--transformation_type',
                    choices=['prop', 'clr', 'alr', 'none'],
                    default='prop',
                    help='data transformation')
parser.add_argument('-to', '--transformation_opts',
                    nargs='+',
                    action=ParseKwargs,
                    help='data transformation options (as key=value)')
parser.add_argument('-ne', '--n_estimators',
                    type=int,
                    default=500,
                    help='n_estimators in scikit-learn RandomForestClassifier')
parser.add_argument('-mf', '--max_features',
                    default='sqrt',
                    help='max_features in scikit-learn RandomForestClassifier')
parser.add_argument('-md', '--max_depth',
                    default=None,
                    help='max_depth in scikit-learn RandomForestClassifier')
parser.add_argument('-ms', '--max_samples',
                    default=None,
                    help='max_samples in scikit-learn RandomForestClassifier')
parser.add_argument('-mss', '--min_samples_split',
                    default=2,
                    help='min_samples_split in scikit-learn RandomForestClassifier')
parser.add_argument('-im', '--importance_measure',
                    choices=['SHAP', 'permutation'],
                    default='SHAP',
                    help='measure of feature importance')
parser.add_argument('-lb', '--balanced',
                    action='store_true',
                    help='use balanced samples for LODO')
args = parser.parse_args()


# Convert max_features to integer, float, or string
if args.max_features.isdigit():
    max_features = int(args.max_features)
else:
    try:
        max_features = float(args.max_features)
    except ValueError:
        max_features = args.max_features

# Make dictionary of transformation options
if args.transformation_opts is None:
    transformation_opts = {}
else:
    transformation_opts = args.transformation_opts

# Dictionary of model hyperparameters
model_opts = {"n_estimators": args.n_estimators,
              "max_features": max_features,
              "max_depth": args.max_depth,
              "max_samples": args.max_samples,
              "min_samples_split": args.min_samples_split}

if 'all' in args.analysis_type:
    analysis_type = ['intra', 'cross', 'lodo']
else:
    analysis_type = args.analysis_type

if 'intra' in analysis_type:
    intra_dataset(data_name=args.data_name,
                  file_suffix=args.file_suffix,
                  num_repetitions=args.num_repetitions,
                  num_folds=args.num_folds,
                  transformation_type=args.transformation_type,
                  transformation_opts=transformation_opts,
                  model_opts=model_opts)
if 'cross' in analysis_type:
    cross_dataset(data_name=args.data_name,
                  file_suffix=args.file_suffix,
                  num_repetitions=args.num_repetitions,
                  transformation_type=args.transformation_type,
                  transformation_opts=transformation_opts,
                  model_opts=model_opts,
                  importance_measure=args.importance_measure)
if 'lodo' in analysis_type:
    LODO(data_name=args.data_name,
         file_suffix=args.file_suffix,
         num_repetitions=args.num_repetitions,
         transformation_type=args.transformation_type,
         transformation_opts=transformation_opts,
         model_opts=model_opts,
         balanced=args.balanced)
