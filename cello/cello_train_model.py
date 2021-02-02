"""
Train a CellO model

If a given dataset's genes do not match the genes expected
by CellO's pre-trained classifiers, then a new model needs to be
trained on the input genes. This script enables this training.

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

from optparse import OptionParser
import dill
import os

from . import load_expression_matrix
import cello

def main():
    usage = "usage: %prog <expression_matrix_file>"
    parser = OptionParser(usage)
    parser.add_option("-d", "--data_type", help="Data type (required). Must be one of: 'TSV', 'CSV', '10x', or 'HDF5'")
    parser.add_option("-a", "--algo", help="Hierarchical classification algorithm to apply (default='IR'). Must be one of: 'IR' - Isotonic regression, 'CLR' - cascaded logistic regression")
    parser.add_option("-c", "--h5_cell_key", help="The key of the dataset within the input HDF5 file specifying which dataset stores the cell ID's.  This argument is only applicable if '-d HDF5' is used")
    parser.add_option("-g", "--h5_gene_key", help="The key of the dataset within the input HDF5 file specifying which dataset stores the gene names/ID's.  This argument is only applicable if '-d HDF5' is used")
    parser.add_option("-e", "--h5_expression_key", help="The key of the dataset within the input HDF5 file specifying which dataset stores the expression matrix.  This argument is only applicable if '-d HDF5' is used")
    parser.add_option("-r", "--rows_cells", action="store_true", help="Use this flag if table is CELLS x GENES rather than GENES x CELLS. Not applicable for 10x input.")
    parser.add_option("-f", "--resource_location", help="Location of resources")
    parser.add_option("-o", "--output_file", help="File in which to write model")
    (options, args) = parser.parse_args()

    data_loc = args[0]
    data_type = options.data_type
    if options.algo:
        algo = options.algo
    else:
        algo = 'IR'
    out_f = options.output_file

    # Get location of CellO resources
    if options.resource_location:
        rsrc_loc = options.resource_location
    else:
        rsrc_loc = os.getcwd()

    # If HDF5 data is provided parse options for dataset keys
    if options.data_type is not None and options.data_type == 'HDF5':
        try:
            assert options.h5_cell_key is not None
            assert options.h5_gene_key is not None
            assert options.h5_expression_key is not None
        except:
            print()
            print("Error. The specified input data is HDF5.  The dataset keys within the HDF5 must be provided via the '-c', '-g', and '-e' arguments.  Please run 'python cello_predict.py -h' for more details.")
            exit()

    # Load data matrix
    ad = load_expression_matrix.load_data(
        data_loc,
        data_type,
        hdf5_expr_key=options.h5_expression_key,
        hdf5_cells_key=options.h5_cell_key,
        hdf5_genes_key=options.h5_gene_key
    )

    # Train model
    mod = cello.train_model(ad, rsrc_loc, algo=algo)

    # Write output
    print("Writing model to {}.".format(out_f))
    with open(out_f, 'wb') as f:
        dill.dump(mod, f)

if __name__ == '__main__':
    main()
