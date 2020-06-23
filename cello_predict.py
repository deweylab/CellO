"""
Run CellO on a gene expression matrix

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

from optparse import OptionParser
from os.path import join
import pandas as pd

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    parser.add_option("-a", "--algo", help="Hierarchical classification algorithm to apply (default='IR'). Must be one of: 'IR' - Isotonic regression, 'CLR' - cascaded logistic regression")
    parser.add_option("-d", "--data_type", help="Data type (required). Must be one of: 'TSV', 'CSV', '10x', or 'HDF5'")
    parser.add_option("-r", "--rows_cells", action="store_true", help="Use this flag if table is CELLS x GENES rather than GENES x CELLS. Not applicable for 10x input.")
    parser.add_option("-u", "--units", help="Units of expression. Must be one of: {COUNTS, CPM, LOG1_CPM, TPM, LOG1_TPM}")
    parser.add_option("-s", "--assay", help="Units of expression. Must be one of: 3_PRIME, FULL_LENGTH")
    parser.add_option("-t", "--train_model", help="If the genes in the input matrix don't match what is expected by the classifier, then train a classifier on the input genes. The model will be saved to <output_prefix>.model.dill")
    parser.add_option("-m", "--model", help="Path to pretrained model file.")
    parser.add_option("-o", "--output_prefix", help="Prefix for all output files. This prefix may contain a path.")
    (options, args) = parser.parse_args()

    data_loc = args[0]
    out_pref = options.output_prefix

    # Load CellO after parsing arguments since CellO takes a while
    # to load the ontologies
    from cello import load_expression_matrix
    from cello import CellO

    # Parse options
    if options.data_type:
        data_type = options.data_type
    else:
        data_type = 'TSV'

    if options.model:
        model_f = options.model
    else:
        model_f = None

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

    # Load data
    print('Loading data from {}...'.format(data_loc))
    ad = load_expression_matrix.load_data(data_loc, data_type)
    print("Loaded data matrix with {} cells and {} genes.".format(
        ad.X.shape[0],
        ad.X.shape[1]
    ))

    results_df, finalized_binary_results_df = CellO.predict(
        ad,
        CellO.COUNTS_UNITS,
        '3_PRIME',
        algo=algo,
        cluster=True,
        res=1.0,
        model_f=model_f
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
