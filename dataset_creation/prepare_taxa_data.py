"""Create datasets with species filtered by statistic."""
import pickle
import argparse
import dataprep


def main(file_name='CRC_taxa.pkl',
         statistic='max_abundance',
         threshold=1e-3, min_reads=1e6):
    """Create dataset with species filtered by statistic."""

    # Load data and put into dictionary with correct order
    print('Loading taxonomic data')
    taxa_dict = dataprep.create_taxa_dict(data_dir='CMD_files') 
    taxa_dict = dataprep.order_dict(taxa_dict)

    # Filter features based on statistic and threshold
    feature_list = dataprep.filter_by_stat(taxa_dict,
                                           statistic,
                                           threshold,
                                           transformation=dataprep.prop)

    # Remove features and low-read samples. Standardize feature tables.
    taxa_dict = dataprep.select_features(taxa_dict, feature_list)
    taxa_dict = dataprep.remove_samples(taxa_dict, taxa_dict, min_reads)
    taxa_dict = dataprep.add_zero_columns(taxa_dict)

    # Save resulting datasets
    print('\nSaving data as {}'.format(file_name))
    with open('../data/{}'.format(file_name), 'wb') as f:
        pickle.dump(taxa_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name',
                        default='CRC_taxa.pkl',
                        help='output file name')
    parser.add_argument('-s', '--statistic',
                        choices=['max_abundance', 'prevalence', 'variance'],
                        default='max_abundance',
                        help='type of statistic to threshold')
    parser.add_argument('-t', '--threshold',
                        type=float,
                        default=1e-3,
                        help='threshold for cutoff')
    parser.add_argument('-m', '--min_reads',
                        type=int,
                        default=1e6,
                        help='minimum number of reads to keep sample')
    args = parser.parse_args()

    main(args.file_name, args.statistic, args.threshold, args.min_reads)
