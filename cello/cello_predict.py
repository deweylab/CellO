"""
Run CellO on a gene expression matrix

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

from optparse import OptionParser
import os
from os.path import join
import pandas as pd
from anndata import AnnData
import dill
import subprocess
import sys
from . import cello
from . import ontology_utils as ou
from . import load_expression_matrix
try:
    import scanpy as sc
except ImportError:
    sys.exit("The 'cello_predict' command line tool requires that scanpy package be installed. To install scanpy, run 'pip install scanpy'.")

# Units keywords
COUNTS_UNITS = 'COUNTS'
CPM_UNITS = 'CPM'
LOG1_CPM_UNITS = 'LOG1_CPM'
TPM_UNITS = 'TPM'
LOG1_TPM_UNITS = 'LOG1_TPM'

# Assay keywords
FULL_LENGTH_ASSAY = 'FULL_LENGTH'
THREE_PRIMED_ASSAY = '3_PRIME'

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
    parser.add_option("-f", "--resource_location", help="Path to CellO resources directory, named 'resources',  which stores gene mappings, pre-trained models, and training sets. If not supplied, CellO will look for 'resources' in the current directory. If resources do not exist at provided location, they will be downloaded automatically.")
    parser.add_option("-m", "--model", help="Path to pretrained model file.")
    parser.add_option("-l", "--remove_anatomical", help="A comma-separated list of terms ID's from the Uberon Ontology specifying which tissues to use to filter results. All cell types known to be resident to the input tissues will be filtered from the results.")
    parser.add_option("-p", "--pre_clustering", help="A TSV file with pre-clustered cells. The first column stores the cell names/ID's (i.e. the column names of the input expression matrix) and the second column stores integers referring to each cluster. The TSV file should not have column names.")
    parser.add_option("-b", "--ontology_term_ids", action="store_true", help="Use the less readable, but more rigorous Cell Ontology term id's in output")
    parser.add_option("-o", "--output_prefix", help="Prefix for all output files. This prefix may contain a path.")
    (options, args) = parser.parse_args()

    data_loc = args[0]
    out_pref = options.output_prefix

    if options.resource_location:
        rsrc_loc = options.resource_location
    else:
        rsrc_loc = os.getcwd()

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
            print()
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

    # Parse the pre-clustered cells
    if options.pre_clustering:
        pre_clustering_f = options.pre_clustering
        cell_to_cluster = {}
        with open(pre_clustering_f, 'r') as f:
            for l in f:
                toks = l.split('\t')
                cell = toks[0].strip()
                clust = int(toks[1].strip())
                cell_to_cluster[cell] = clust
    else:
        cell_to_cluster = None
        
    try:
        assert options.units
    except:
        print("Error. Please specify units using the '-u' ('--units') option.")
        print("For more details, run with '-h' ('--help') option.")
        return
    units = options.units
    assay = options.assay

    # One last argument to parse that relies on the Cell Ontology itself
    remove_anatomical_subterms = None
    if options.remove_anatomical:
        remove_anatomical_subterms = options.remove_anatomical.split(',')
        for term in remove_anatomical_subterms:
            try:
                assert term in ou.cell_ontology().id_to_term
            except AssertionError:
                print()
                print('Error. For argument --remove_anatomical (-l), the term "{}" was not found in the Uberon Ontology.'.format(term))
                exit()

    # Create log directory
    log_dir = '{}.log'.format(out_pref)
    subprocess.run('mkdir {}'.format(log_dir), shell=True)

    # Load data
    print('Loading data from {}...'.format(data_loc))
    ad = load_expression_matrix.load_data(
        data_loc, 
        data_type,
        hdf5_expr_key=options.h5_expression_key,
        hdf5_cells_key=options.h5_cell_key,
        hdf5_genes_key=options.h5_gene_key
    )
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
        model = cello._retrieve_pretrained_model(ad, algo, rsrc_loc)
        if model is None:
            if options.train_model:
                model = cello.train_model(ad, rsrc_loc, algo=algo, log_dir=log_dir)
                out_model_f = '{}.model.dill'.format(out_pref)
                print('Writing trained model to {}'.format(out_model_f))
                with open(out_model_f, 'wb') as f:
                    dill.dump(model, f)

    if model is None:
        print()
        print("Error. The genes present in data matrix do not match those expected by any of the pre-trained classifiers.")
        print("Please train a classifier on this input gene set by either using the cello_train_model.py program or by running cello_classify with the '-t' flag.")
        exit()        

    results_df, finalized_binary_results_df, ms_results_df = run_cello(
        ad,
        units,
        model,
        assay=assay,
        algo=algo,
        cluster=True,
        cell_to_clust=cell_to_cluster,
        log_dir=log_dir,
        res=1.0,
        remove_anatomical_subterms=remove_anatomical_subterms
    )

    # Convert to human-readable ontology terms
    if not options.ontology_term_ids:
        results_df.columns = [
            ou.cell_ontology().id_to_term[x].name
            for x in results_df.columns
        ]
        finalized_binary_results_df.columns = [
            ou.cell_ontology().id_to_term[x].name
            for x in finalized_binary_results_df.columns
        ]
        ms_results_df['most_specific_cell_type'] = [
            ou.cell_ontology().id_to_term[x].name
            for x in ms_results_df['most_specific_cell_type']
        ]

    # Write output
    out_f = '{}.probability.tsv'.format(out_pref)
    print("Writing classifier probabilities to {}...".format(out_f))
    results_df.to_csv(out_f, sep='\t')

    out_f = '{}.binary.tsv'.format(out_pref)
    print("Writing binarized classifications to {}...".format(out_f))
    finalized_binary_results_df.to_csv(out_f, sep='\t')

    out_f = '{}.most_specific.tsv'.format(out_pref)
    print("Writing most-specific cell types to {}...".format(out_f))
    ms_results_df.to_csv(out_f, sep='\t')


def run_cello(
        ad,
        units,
        mod,
        assay='3_PRIME',
        algo='IR',
        cluster=True,
        cell_to_clust=None,
        log_dir=None,
        res=1.0,
        remove_anatomical_subterms=None
    ):
    # Get units into log(TPM+1)
    if assay == FULL_LENGTH_ASSAY:
        if units in set([COUNTS_UNITS, CPM_UNITS, LOG1_CPM_UNITS]):
            print('Error. The input units were specified as {}'.format(units),
                'but the assay was specified as {}.'.format(assay),
                'To run classification, please input expression matrix in ',
                'units of either LOG1_TPM or log(TPM+1) for this assay type.')
            exit()
    if units == COUNTS_UNITS:
        print('Normalizing counts...')
        sc.pp.normalize_total(ad, target_sum=1e6)
        sc.pp.log1p(ad)
        print('done.')
    elif units in set([CPM_UNITS, TPM_UNITS]):
        sc.pp.log1p(ad)

    if cluster and ad.X.shape[0] > 50 and cell_to_clust is None:
        # Run clustering
        sc.pp.pca(ad)
        sc.pp.neighbors(ad)
        sc.tl.leiden(ad, resolution=res)
        clust_key = 'leiden'
    elif cluster and cell_to_clust is not None:
        # Clusters are already provided
        ad.obs['cluster'] = [
            cell_to_clust[cell]
            for cell in ad.obs.index
        ]
        clust_key = 'cluster'
    else:
        # Do not run clustering
        clust_key = None

    results_df, finalized_binary_results_df, ms_results_df = cello.predict(
        ad,
        mod,
        algo=algo,
        clust_key=clust_key,
        log_dir=log_dir,
        remove_anatomical_subterms=remove_anatomical_subterms
    )

    return results_df, finalized_binary_results_df, ms_results_df

if __name__ == '__main__':
    main()
