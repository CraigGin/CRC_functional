"""Create dataset with gene families filtered by statistic."""
import pickle
import argparse
import dataprep


def main(file_prefix='CRC',
         statistic='max_abundance',
         thr_list=[1e-4],
         min_reads=1e6):
    """Create dataset with gene families filtered by statistic."""
    
    # Load data and put into dictionary with correct order
    print('Loading gene family data')
    full_dict = dataprep.create_gf_dict(data_dir='CMD_files')
    full_dict = dataprep.order_dict(full_dict)
    
    # Find gene families with statistic over threshold in most studies
    for thr in thr_list:
        keep_list = dataprep.filter_by_stat(full_dict, statistic,
                                            thr, transformation=dataprep.prop)
        disease_dict = dataprep.select_features(full_dict, keep_list)

        # Load taxonomic data and put into dictionary
        print('Loading taxonomic data')
        taxa_dict = dataprep.create_taxa_dict(data_dir='CMD_files') 
        taxa_dict = dataprep.order_dict(taxa_dict)

        # Remove samples with read count less than min_reads
        print('Removing Samples')
        disease_dict = dataprep.remove_samples(disease_dict,
                                               taxa_dict,
                                               min_reads)

        # Make it so all study matrices have the same columns
        disease_dict = dataprep.add_zero_columns(disease_dict)

        # Check final sizes
        print('Size of final abundance matrices:')
        for study in disease_dict:
            print('{}: {}'.format(study, disease_dict[study]['X'].shape))

        output_file = '{}_{}_{}.pkl'.format(file_prefix, statistic, thr)
        print('\nSaving data as {}'.format(output_file))
        with open('../data/{}'.format(output_file), 'wb') as f:
            pickle.dump(disease_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_prefix',
                        default='CRC',
                        help='prefix for output file name')
    parser.add_argument('--statistic',
                        choices=['max_abundance', 'prevalence', 
                                 'variance'],
                        default='max_abundance',
                        help='type of statistic to threshold')
    parser.add_argument('-t', '--threshold',
                        nargs='+',
                        type=float,
                        default=[1e-4],
                        help='thresholds for cutoff')
    parser.add_argument('-m', '--min_reads',
                        type=int,
                        default=1e6,
                        help='minimum number of reads to keep sample')
    args = parser.parse_args()

    main(args.file_prefix, args.statistic, args.threshold, args.min_reads)
