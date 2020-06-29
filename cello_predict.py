"""
Run CellO on a gene expression matrix

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

from optparse import OptionParser
from os.path import join
import pandas as pd
import dill

def main():
    usage = "%prog [options] input_file" 
    parser = OptionParser(usage=usage)
    parser.add_option("-a", "--algo", help="Hierarchical classification algorithm to apply (default='IR'). Must be one of: 'IR' - Isotonic regression, 'CLR' - cascaded logistic regression")
    parser.add_option("-d", "--data_type", help="Data type (required). Must be one of: 'TSV', 'CSV', '10x', or 'HDF5'. Note: if 'HDF5' is used, then arguments must be provided to the h5_cell_key, h5_gene_key, and h5_expression_key parameters.")
    parser.add_option("-c", "--h5_cell_key", help="The key of the dataset within the input HDF5 file specifying which dataset stores the cell ID's.  This argument is only applicable if '-d HDF5' is used")
    parser.add_option("-g", "--h5_gene_key", help="The key of the dataset within the input HDF5 file specifying which dataset stores the gene names/ID's.  This argument is only applicable if '-d HDF5' is used")
    parser.add_option("-e", "--h5_expression_key", help="The key of the dataset within the input HDF5 file specifying which dataset stores the expression matrix.  This argument is only applicable if '-d HDF5' is used")
    parser.add_option("-r", "--rows_cells", action="store_true", help="Use this flag if expression matrix is organized as CELLS x GENES rather than GENES x CELLS. Not applicable when '-d 10x' is used.")
    parser.add_option("-u", "--units", help="Units of expression. Must be one of: 'COUNTS', 'CPM', 'LOG1_CPM', 'TPM', 'LOG1_TPM'")
    parser.add_option("-s", "--assay", help="Sequencing assay. Must be one of: '3_PRIME', 'FULL_LENGTH'")
    parser.add_option("-t", "--train_model", action="store_true", help="If the genes in the input matrix don't match what is expected by the classifier, then train a classifier on the input genes. The model will be saved to <output_prefix>.model.dill")
    parser.add_option("-m", "--model", help="Path to pretrained model file.")
    parser.add_option("-o", "--output_prefix", help="Prefix for all output files. This prefix may contain a path.")
    (options, args) = parser.parse_args()

    data_loc = args[0]
    out_pref = options.output_prefix

    # Input validation
    if options.model is not None and options.train_model is not None:
        print("Warning! Option 'train_model' was used along with the")
        print("option 'model'.  These are conflicting arguments. ")
        print("CellO will use the model file provided to 'model' ")
        print("instead of training a new one.")
        options.train_model = False

    if options.data_type is not None and options.data_type == 'HDF5':
        try:
            assert options.h5_cell_key is not None
            assert options.h5_gene_key is not None
            assert options.h5_expression_key is not None
        except:
            print("Error. The specified input data is HDF5.  The dataset keys within the HDF5 must be provided via the '-c', '-g', and '-e' arguments.  Please run 'python cello_predict.py -h' for more details.")
            exit()

    # Parse options
    if options.data_type:
        data_type = options.data_type
    else:
        print("Warning! A data format was not specified with the '-d' option. Assuming that input is a tab-separated-value (TSV) file.")
        data_type = 'TSV'

    if options.algo:
        algo = options.algo
    else:
        algo = 'IR'

    try:
        assert options.units
    except:
        print("Please specify units using the '-u' ('--units') option.")
        print("For more details, run with '-h' ('--help') option.")
        return
    units = options.units
    assay = options.assay

    # Load CellO after parsing arguments since CellO takes a while
    # to load the ontologies
    from utils import load_expression_matrix
    import CellO

    # Load data
    print('Loading data from {}...'.format(data_loc))
    ad = load_expression_matrix.load_data(data_loc, data_type)
    print("Loaded data matrix with {} cells and {} genes.".format(
        ad.X.shape[0],
        ad.X.shape[1]
    ))


    # Load or train model
    if options.model:
        model_f = options.model
        print('Loading model from {}...'.format(model_f))
        with open(model_f, 'rb') as f:
            model=dill.load(f)
    else:
        # Load or train a model
        model = CellO._retrieve_pretrained_model(ad, algo)
        if model is None:
            if options.train_model:
                model = CellO.train_model(ad, algo=algo)
                out_model_f = '{}.model.dill'.format(out_pref)
                print('Writing trained model to {}'.format(out_model_f))
                with open(out_model_f, 'wb') as f:
                    dill.dump(model, f)

    results_df, finalized_binary_results_df = CellO.predict(
        ad,
        CellO.COUNTS_UNITS,
        model,
        assay='3_PRIME',
        algo=algo,
        cluster=True,
        res=1.0
    )

    # Write output
    out_f = '{}.probability.tsv'.format(out_pref)
    print("Writing classifier probabilities to {}...".format(out_f))
    results_df.to_csv(out_f, sep='\t')

    out_f = '{}.binary.tsv'.format(out_pref)
    print("Writing binarized classifications to {}...".format(out_f))
    finalized_binary_results_df.to_csv(out_f, sep='\t')

if __name__ == '__main__':
    main()
