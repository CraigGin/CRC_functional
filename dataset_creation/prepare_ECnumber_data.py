"""Create dataset with gene families grouped by EC number."""
import pickle
import argparse
import dataprep


def create_EC_dict(uniprot_file):
    """Make a dictionary with only gene families with an EC number."""

    # Load gene family info from Uniprot
    print('Loading UniProt annotations\n')
    with open(uniprot_file, 'rb') as f:
        annot_dict = pickle.load(f)

    # Keep only entries with EC number in description
    EC_dict = {uid: annot_dict[uid] for uid in annot_dict
               if 'EC=' in annot_dict[uid]['description']}
    print('{} gene families have an EC number in Uniprot'.format(len(EC_dict)))

    # Add a field called 'EC' that contains the EC number, remove semicolons
    for uid in EC_dict:
        for desc in EC_dict[uid]['description'].split():
            if 'EC=' in desc:
                EC_dict[uid]['EC'] = desc[3:].replace(';', '')
                continue

    # Get rid of entries that are not fully resolved
    EC_dict = {uid: EC_dict[uid] for uid in EC_dict
               if '-' not in EC_dict[uid]['EC']}
    print('{} gene families have a fully resolved '.format(len(EC_dict)) +
          'EC number')

    print('Saving EC_dict.pkl\n')
    with open('EC_dict.pkl', 'wb') as f:
        pickle.dump(EC_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(output_file='CRC_ECnumber.pkl',
         uniprot_file='uniprot.pkl',
         min_reads=1e6):
    """Create dataset with gene families grouped by EC number."""
    
    # Load data and put into dictionary with correct order
    print('Loading gene-family data')
    disease_dict = dataprep.create_gf_dict(data_dir='CMD_files')
    disease_dict = dataprep.order_dict(disease_dict)

    # Create dictionary with EC numbers
    create_EC_dict(uniprot_file)

    # Load dictionary
    with open('EC_dict.pkl', 'rb') as f:
        annot_dict = pickle.load(f)

    # Get set of unique EC numbers
    EC_numbers = {annot_dict[uid]['EC'] for uid in annot_dict}
    print('There are {} unique EC numbers\n'.format(len(EC_numbers)))

    # Dictionary with list of gene families for each unique EC number
    column_dict = {EC: ['UniRef90_{}'.format(gf) for gf in annot_dict
                        if annot_dict[gf]['EC'] == EC]
                   for EC in EC_numbers}

    disease_dict = dataprep.group_features(disease_dict, column_dict)

    # Load taxonomic data and put into dictionary
    print('Loading taxonomic data')
    taxa_dict = dataprep.create_taxa_dict(data_dir='CMD_files') 
    taxa_dict = dataprep.order_dict(taxa_dict)

    # Remove samples with read count less than min_reads
    print('Removing Samples')
    disease_dict = dataprep.remove_samples(disease_dict, taxa_dict, min_reads)

    print('Saving data as {}'.format(output_file))
    with open('../data/{}'.format(output_file), 'wb') as f:
        pickle.dump(disease_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-uf', '--uniprot_file',
                        default='uniprot_annot.pkl',
                        help='name of file containing uniprot annotations')
    parser.add_argument('-of', '--output_file',
                        default='CRC_ECnumber.pkl',
                        help='output file name')
    parser.add_argument('-m', '--min_reads',
                        type=int,
                        default=1e6,
                        help='minimum number of reads to keep sample')
    args = parser.parse_args()

    main(args.output_file, args.uniprot_file, args.min_reads)
