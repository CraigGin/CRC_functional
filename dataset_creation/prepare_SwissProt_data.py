"""Create dataset with Swiss-Prot gene-families."""
import pickle
import argparse
import dataprep


def main(output_file='CRC_SwissProt.pkl',
         uniprot_file='uniprot.pkl',
         min_reads=1e6):
    """Create dataset with Swiss-Prot gene-families."""
    
    # Load data and put into dictionary with correct order
    print('Loading gene-family data')
    disease_dict = dataprep.create_gf_dict(data_dir='CMD_files')
    disease_dict = dataprep.order_dict(disease_dict)

    # Load gene family info from Uniprot
    print('Loading UniProt annotations\n')
    with open(uniprot_file, 'rb') as f:
        annot_dict = pickle.load(f)

    gf_SwissProt = ['UniRef90_{}'.format(uid)
                    for uid in annot_dict.keys()
                    if annot_dict[uid]['is_reviewed']]

    disease_dict = dataprep.select_features(disease_dict, gf_SwissProt)

    # Load taxonomic data and put into dictionary
    print('Loading taxonomic data')
    taxa_dict = dataprep.create_taxa_dict(data_dir='CMD_files') 
    taxa_dict = dataprep.order_dict(taxa_dict)

    # Remove samples with read count less than min_reads
    print('Removing Samples')
    disease_dict = dataprep.remove_samples(disease_dict, taxa_dict, min_reads)

    # Make it so all study matrices have the same columns
    disease_dict = dataprep.add_zero_columns(disease_dict)

    print('Saving data as {}'.format(output_file))
    with open('../data/{}'.format(output_file), 'wb') as f:
        pickle.dump(disease_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-uf', '--uniprot_file',
                        default='uniprot.pkl',
                        help='name of file containing uniprot annotations')
    parser.add_argument('-of', '--output_file',
                        default='CRC_SwissProt.pkl',
                        help='output file name')
    parser.add_argument('-m', '--min_reads',
                        type=int,
                        default=1e6,
                        help='minimum number of reads to keep sample')
    args = parser.parse_args()

    main(args.output_file, args.uniprot_file, args.min_reads)
