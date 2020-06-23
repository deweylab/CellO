"""
Train a CellO model

If a given dataset's genes do not match the genes expected
by CellO's pre-trained classifiers, then a new model needs to be
trained on the input genes. This script enables this training.

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

from optparse import OptionParser
import dill

def main():
    usage = "usage: %prog <expression_matrix_file>"
    parser = OptionParser(usage)
    parser.add_option("-d", "--data_type", help="Data type (required). Must be one of: 'TSV', 'CSV', '10x', or 'HDF5'")
    parser.add_option("-a", "--algo", help="Data type (required). Must be one of: 'TSV', 'CSV', '10x', or 'HDF5'")
    parser.add_option("-r", "--rows_cells", action="store_true", help="Use this flag if table is CELLS x GENES rather than GENES x CELLS. Not applicable for 10x input.")
    parser.add_option("-o", "--output_file", help="File in which to write model")
    (options, args) = parser.parse_args()

    # Load CellO after parsing arguments since CellO takes a while
    # to load the ontologies
    from cello import load_expression_matrix
    from cello import CellO

    data_loc = args[0]
    data_type = options.data_type
    if options.algo:
        algo = options.algo
    else:
        algo = 'IR'
    out_f = options.output_file

    # Load data matrix
    ad = load_expression_matrix.load_data(
        data_loc,
        data_type
    )

    # Train model
    mod = CellO.train_model(ad, algo=algo)

    # Write output
    print("Writing model to {}.".format(out_f))
    with open(out_f, 'wb') as f:
        dill.dump(mod, f)

if __name__ == '__main__':
    main()
