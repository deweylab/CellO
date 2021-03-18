"""
The CellO API

Authors: Matthew Bernstein <mbernstein@morgridge.org>
"""

from optparse import OptionParser
from os.path import join
import pandas as pd
from anndata import AnnData
import h5py
import dill
from scipy.io import mmread
import numpy as np
import json
from collections import defaultdict
import sys
import os

from . import the_ontology
from . import load_expression_matrix
from . import load_training_data
from . import download_resources
from . import ontology_utils as ou
from . import models
from .models import model
from .graph_lib.graph import DirectedAcyclicGraph

# Units keywords
COUNTS_UNITS = 'COUNTS'
CPM_UNITS = 'CPM'
LOG1_CPM_UNITS = 'LOG1_CPM'
TPM_UNITS = 'TPM'
LOG1_TPM_UNITS = 'LOG1_TPM'

# Assay keywords
FULL_LENGTH_ASSAY = 'FULL_LENGTH'
THREE_PRIMED_ASSAY = '3_PRIME'

UNITS = 'log_tpm'

ALGO_TO_INTERNAL = {
    'IR': 'isotonic_regression',
    'CLR': 'cdc'
}
ALGO_TO_PARAMS = {
    'IR': {
        "assert_ambig_neg": False,
        "binary_classifier_algorithm": "logistic_regression",
        "binary_classifier_params": {
            "penalty": "l2",
            "penalty_weight": 0.0006,
            "solver": "liblinear",
            "intercept_scaling": 1000.0,
            "downweight_by_class": True
        }
    },
    'CLR': {
        "assert_ambig_neg": False,
        "binary_classifier_algorithm": "logistic_regression",
        "binary_classifier_params": {
            "penalty": "l2",
            "penalty_weight": 0.001,
            "solver": "liblinear",
            "intercept_scaling": 1000.0,
            "downweight_by_class": True
        }
    }
}
PREPROCESSORS = ['pca']
PREPROCESSOR_PARAMS = [{
    "n_components": 3000
}]

QUALIFIER_TERMS = set([
    'CL:2000001',   # peripheral blood mononuclear cell
    'CL:0000081',   # blood cell
    'CL:0000080',   # circulating cell
    'CL:0002321'    # embryonic cell
])

def train_model(ad, rsrc_loc, algo='IR', log_dir=None):
    """
    Train a CellO model based on the genes of an 
    input dataset.

    Parameters
    ----------

    ad : AnnData object
        Expression matrix of n cells by m genes

    algo : String
        The name of the algorithm used to train the model. 'IR' 
        trains a model using isotonic regression. 'CLR' trains
        a model using cascaded logistic regression.

    rsrc_loc: String
        The location of the "resources" directory downloaded
        via the ''

    log_dir : String
        Path to a directory in which to write logging information

    Returns
    -------
    A trained CellO model
    """
    _download_resources(rsrc_loc)

    genes = ad.var.index

    # Load the training data
    r = load_training_data.load(UNITS, rsrc_loc)
    og = r[0]
    label_graph = r[1]
    label_to_name = r[2]
    the_exps = r[3]
    exp_to_index = r[4]
    exp_to_labels = r[5]
    exp_to_tags = r[6]
    exp_to_study = r[7]
    study_to_exps = r[8]
    exp_to_ms_labels = r[9]
    X = r[10]
    all_genes = r[11]

    # Match genes in test data to those in training
    # data
    train_genes, gene_to_indices = _match_genes(
        genes, 
        all_genes, 
        rsrc_loc, 
        log_dir=log_dir
    )

    # Take a subset of the columns for the training-genes. Note
    # that if a given gene in the test set maps to multiple training
    # genes, then we sum over the training genes. 
    X_train = []
    for gene in train_genes:
        indices = gene_to_indices[gene]
        X_train.append(np.sum(X[:,indices], axis=1))
    X_train = np.array(X_train).T
    assert X_train.shape[1] == len(train_genes)

    # Train the model on these genes
    print('Training model...')
    mod = model.train_model(
        ALGO_TO_INTERNAL[algo],
        ALGO_TO_PARAMS[algo],
        X_train,
        the_exps,
        exp_to_labels,
        label_graph,
        item_to_group=exp_to_study,
        features=train_genes,
        preprocessor_names=PREPROCESSORS,
        preprocessor_params=PREPROCESSOR_PARAMS
    )
    print('done.')
    return mod


def load_training_set():
    r = load_training_data.load(UNITS)
    return r


def predict(
        ad,
        mod,
        algo='IR',
        clust_key='leiden',
        log_dir=None,
        remove_anatomical_subterms=None,
        rsrc_loc=None
    ):
    """
    Classify cell types for a given expression matrix.

    Parameters
    ----------
    ad : AnnData object
        Expression matrix of n cells by m genes

    mod: Model, default: None
        A trained classification model. If None, use a 
        pre-trained default model on all of the genes, or,
        if 'train' is True, then train a new model.

    algo: String, optional, default: 'IR'
        The hierarchical classification algorithm to use. Must be one of {'IR', 'CLR'}. 
        These keywords refer to:
            IR: isotonic regression correction
            CLR: cascaded logistic regression

    clust_key: String, default: 'leiden'
        The key name in the observation annotation '.obs' that denotes
        the cluster identity of each cell.

    rsrc_loc: String, default: current directory
        Location of the CellO resources. If they are not located
        at this path, they will downloaded automatically.

    Returns
    -------
    probabilities, binary_classificaitons : two Pandas DataFrame
        objects, where the first dataframe stores the raw probabilities
        for each cell type across all input cells and the second stores
        the binarized (1 or 0) classifications across all cell types
        for each input cell.
    """

    # Set resource location to current working directory
    if rsrc_loc is None:
        rsrc_loc = os.getcwd()

    # Download resources if they don't exist
    _download_resources(rsrc_loc)

    # Check that model is combatible with data
    is_compatible = check_compatibility(ad, mod)
    if not is_compatible:
        print("Error. The genes present in data matrix do not match those expected by the classifier.")
        print("Please train a classifier on this input gene set by either using the cello_train_model.py ")
        print("program or by running cello_classify with the '-t' flag.")
        exit()


    # Compute raw classifier probabilities
    results_df, cell_to_clust = _raw_probabilities(
        ad,
        mod,
        algo=algo,
        clust_key=clust_key,
        log_dir=log_dir
    )

    # Filter by anatomical entity
    if remove_anatomical_subterms is not None:
        print("Filtering predictions for cells found in:\n{}".format(
            "\n".join([
                "{} ({})".format(
                    ou.cell_ontology().id_to_term[term].name,
                    term
                )
                for term in remove_anatomical_subterms
            ])
        ))
        results_df = _filter_by_anatomical_entity(
            results_df, 
            remove_subterms_of=remove_anatomical_subterms
        )

    # Binarize the output probabilities
    threshold_df = _retrieve_empirical_thresholds(ad, algo, rsrc_loc)
    label_graph = _retrieve_label_graph(rsrc_loc)
    binary_results_df = _binarize_probabilities(
        results_df,
        threshold_df,
        label_graph
    )

    # Select one most-specific cell type if there are 
    # more than one
    finalized_binary_results_df, ms_results_df = _select_one_most_specific(
        binary_results_df,
        results_df,
        threshold_df,
        label_graph,
        precision_thresh=0.0
    )

    # Map cluster predictions back to their cells
    if cell_to_clust is not None:
        results_da = [
            results_df.loc[cell_to_clust[cell]]
            for cell in ad.obs.index
        ]
        results_df = pd.DataFrame(
            data=results_da,
            index=ad.obs.index,
            columns=results_df.columns
        )

        finalized_binary_results_da = [
            finalized_binary_results_df.loc[cell_to_clust[cell]]
            for cell in ad.obs.index
        ]
        finalized_binary_results_df = pd.DataFrame(
            data=finalized_binary_results_da,
            index=ad.obs.index,
            columns=finalized_binary_results_df.columns
        )

        ms_results_da = [
            ms_results_df.loc[cell_to_clust[cell]]
            for cell in ad.obs.index
        ]
        ms_results_df = pd.DataFrame(
            data=ms_results_da,
            index=ad.obs.index,
            columns=ms_results_df.columns
        )
    return results_df, finalized_binary_results_df, ms_results_df


def _retrieve_pretrained_model(ad, algo, rsrc_loc):
    _download_resources(rsrc_loc)

    print('Checking if any pre-trained model is compatible with this input dataset...')
    pretrained_ir = [
        'ir.dill',
        'ir.10x.dill',
    ]
    pretrained_clr = [
        'clr.dill',
        'clr.10x.dill'
    ]
    mod = None
    assert algo in ALGO_TO_INTERNAL.keys() 
    if algo == 'IR':
        for model_fname in pretrained_ir:
            model_f = join(
                rsrc_loc,
                "resources",
                "trained_models", 
                model_fname
            )
            with open(model_f, 'rb') as f:
                mod = dill.load(f)  
            feats = mod.classifier.features
            if frozenset(feats) == frozenset(ad.var.index):
                return mod
    elif algo == 'CLR':
        for model_fname in pretrained_clr:
            model_f = join(
                rsrc_loc,
                "resources",
                "trained_models", 
                model_fname
            )
            with open(model_f, 'rb') as f:
                mod = dill.load(f)
            feats = mod.classifier.features
            if frozenset(feats) == frozenset(ad.var.index):
                return mod
    print('Could not find compatible pre-trained model.')
    return None


def _download_resources(rsrc_loc):
    if not os.path.isdir(join(rsrc_loc, "resources")):
        msg = """
        Could not find the CellO resources directory called
        'resources' in '{}'. Will download resources to current 
        directory.
        """.format(rsrc_loc)
        print(msg)
        download_resources.download(rsrc_loc)
    else:
        print("Found CellO resources at '{}'.".format(join(rsrc_loc, 'resources')))


def retreive_pretrained_model_from_local(ad, model_dir):
    """
    Search a local directory that may store custom, pre-trained
    models and return the first model whose expected genes match
    that of the input dataset.

    This is a helper function for managing large collections
    of pre-trained models that may be used for running on diverse
    datasets.

    Parameters
    ----------

    ad : AnnData object
        Expression matrix of n cells by m genes

    model_dir: String
        Path to a directory storing a set of pre-trained models
        encoded as dill files.

    Returns
    -------
    The first model object within model_dir that is compatible
    with the input dataset. Returns None if no model was found.
    """
    for model_fname in os.listdir(model_dir):
        model_f = join(model_dir, model_fname)
        with open(model_f, 'rb') as f:
            mod = dill.load(f)
        feats = mod.classifier.features
        if frozenset(feats) < frozenset(ad.var.index):
            print("Found compatible model in file: ", model_f)
            return mod
    return None

     
def check_compatibility(ad, mod):
    return frozenset(mod.classifier.features) <= frozenset(ad.var.index)


def _raw_probabilities(
        ad, 
        mod,
        algo='IR', 
        clust_key='leiden',
        log_dir=None
    ):
    assert check_compatibility(ad, mod)

    # Shuffle columns to be in accordance with model
    features = mod.classifier.features
    ad = ad[:,features]

    # Cluster
    if clust_key:
        if clust_key not in ad.obs.columns:
            sys.exit(
                """
                Error. Cluster key name {} was not found in the AnnData 
                object's '.obs' variable.
                """.format(clust_key)
            )

        ad_clust = _combine_by_cluster(ad)
        # If there's only one cluster, expand dimensions of expression
        # matrix. AnnData shrinks it, so we need to keep it as a Numpy
        # array.
        if len(ad_clust.X.shape) == 1:
            expr = np.expand_dims(ad_clust.X, 0)
        else:
            expr = ad_clust.X
        conf_df, score_df = mod.predict(expr, ad_clust.obs.index)
        cell_to_clust = {
            cell: str(clust)
            for cell, clust in zip(ad.obs.index, ad.obs[clust_key])
        }
    else:
        cell_to_clust = None
        conf_df, score_df = mod.predict(ad.X, ad.obs.index)
    return conf_df, cell_to_clust


def _aggregate_expression(X):
    """
    Given a matrix of log(TPM+1) where rows correspond to cells
    and columns correspond to genes, aggregate the counts
    to form a psuedo-bulk expression profile.
    """
    X = (np.exp(X)-1) / 1e6
    x_clust = np.sum(X, axis=0)
    sum_x_clust = float(sum(x_clust))
    x_clust = np.array([x/sum_x_clust for x in x_clust])
    x_clust *= 1e6
    x_clust = np.log(x_clust+1)
    return x_clust


def _combine_by_cluster(ad, clust_key='leiden'):
    """
    Given a new AnnData object, we want to create a new object
    where each element isn't a cell, but rather is a cluster.
    """
    clusters = []
    X_mean_clust = []
    for clust in sorted(set(ad.obs[clust_key])):
        cells = ad.obs.loc[ad.obs[clust_key] == clust].index
        X_clust = ad[cells,:].X
        x_clust = _aggregate_expression(X_clust)
        X_mean_clust.append(x_clust)
        clusters.append(str(clust))
    X_mean_clust = np.array(X_mean_clust)
    ad_mean_clust = AnnData(
        X=X_mean_clust,
        var=ad.var,
        obs=pd.DataFrame(
            data=clusters,
            index=clusters
        )
    )
    return ad_mean_clust


def _retrieve_empirical_thresholds(ad, algo, rsrc_loc):
    print('Checking if any pre-trained model is compatible with this input dataset...')
    pretrained_ir = [
        ('ir.dill', 'ir.all_genes_thresholds.tsv'), 
        ('ir.10x.dill', 'ir.10x_genes_thresholds.tsv') 
    ]
    pretrained_clr = [
        ('clr.dill', 'clr.all_genes_thresholds.tsv'),
        ('clr.10x.dill', 'clr.10x_genes_thresholds.tsv')
    ]
    mod = None
    max_genes_common = 0
    best_thresh_f = None
    if algo == 'IR':
        for model_fname, thresh_fname in pretrained_ir:
            model_f = join(
                rsrc_loc,
                "resources",
                "trained_models", 
                model_fname
            )
            with open(model_f, 'rb') as f:
                mod = dill.load(f)  
            feats = mod.classifier.features
            # Compute the fraction of model-features that are in the input dataset
            matched_genes, _ = _match_genes(ad.var.index, feats, rsrc_loc, verbose=False)
            common = len(frozenset(feats) & frozenset(matched_genes)) / len(feats)
            if common >= max_genes_common:
                max_genes_common = common
                best_thresh_f = join(
                    rsrc_loc,
                    "resources",
                    "trained_models", 
                    thresh_fname
                )
    elif algo == 'CLR':
        for model_fname, thresh_fname in pretrained_clr:
            model_f = join(
                rsrc_loc,
                "resources",
                "trained_models", 
                model_fname
            )
            with open(model_f, 'rb') as f:
                mod = dill.load(f)
            feats = mod.classifier.features
            matched_genes, _ = _match_genes(ad.var.index, feats, rsrc_loc, verbose=False)
            common = len(frozenset(feats) & frozenset(matched_genes)) / len(feats)
            if common >= max_genes_common:
                max_genes_common = common
                best_thresh_f = join(
                    rsrc_loc,
                    "resources", 
                    "trained_models", 
                    thresh_fname
                )
    print('Using thresholds stored in {}'.format(best_thresh_f))
    thresh_df = pd.read_csv(best_thresh_f, sep='\t', index_col=0)
    return thresh_df


def _retrieve_label_graph(rsrc_loc):
    labels_f = join(
        rsrc_loc,
        "resources",
        "training_set", 
        "labels.json"
    )
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)
        source_to_targets= labels_data['label_graph']
        exp_to_labels = labels_data['labels']
    label_graph = DirectedAcyclicGraph(source_to_targets)
    return label_graph


def _filter_by_anatomical_entity(
        results_df,
        remove_subterms_of 
    ):
    labels = set(results_df.columns)
    all_subterms = set()
    for term in remove_subterms_of:
        subterms = ou.cell_ontology().recursive_relationship(
            term, 
            ['inv_is_a', 'inv_part_of', 'inv_located_in']
        )
        labels -= subterms
    labels = sorted(labels)
    results_df = results_df[labels]
    return results_df


def _binarize_probabilities(
        results_df, 
        decision_df, 
        label_graph
    ):
    # Map each label to its empirical threshold
    label_to_thresh = {
        label: decision_df.loc[label]['threshold']
        for label in decision_df.index
    }

    print('Binarizing classifications...')
    label_to_descendents = {
        label: label_graph.descendent_nodes(label)
        for label in label_graph.get_all_nodes()
    }

    da = []
    the_labels = sorted(set(results_df.columns) & set(label_to_thresh.keys()))
    for exp_i, exp in enumerate(results_df.index):
        if (exp_i+1) % 100 == 0:
            print('Processed {} samples.'.format(exp_i+1))
        # Map each label to its classification-score 
        label_to_conf = {
            label: results_df.loc[exp][label]
            for label in results_df.columns
        }
        # Compute whether each label is over its threshold
        label_to_is_above = {
            label: int(conf > label_to_thresh[label])
            for label, conf in label_to_conf.items()
            if label in the_labels
        }
        label_to_bin= {
            label: is_above
            for label, is_above in label_to_is_above.items()
        }
        # Propagate the negative predictions to all descendents
        for label, over_thresh in label_to_is_above.items():
            if not bool(over_thresh):
                desc_labels = label_to_descendents[label]
                for desc_label in set(desc_labels) & set(label_to_bin.keys()):
                    label_to_bin[desc_label] = int(False)
        da.append([
            label_to_bin[label]
            for label in the_labels
        ])
    df = pd.DataFrame(
        data=da,
        index=results_df.index,
        columns=the_labels
    )
    return df


def _select_one_most_specific(
        binary_results_df, 
        results_df, 
        decision_df,
        label_graph, 
        precision_thresh=0.0
    ):
    # Parse the decision-thresholds table
    label_to_f1 = {
        label: decision_df.loc[label]['F1-score']
        for label in decision_df.index
    }
    label_to_prec = {
        label: decision_df.loc[label]['precision']
        for label in decision_df.index
    }
    label_to_thresh = {
        label: decision_df.loc[label]['empirical_threshold']
        for label in decision_df.index
    }

    # Map each label to its ancestors
    label_to_ancestors = {
        label: label_graph.ancestor_nodes(label)
        for label in label_graph.get_all_nodes()
    }

    # Filter labels according to empiracle precision
    hard_labels = set([
        label
        for label, prec in label_to_prec.items()
        if prec < precision_thresh
    ])
    
    # Map each experiment to its predicted terms
    print('Mapping each sample to its predicted labels...')
    consider_labels  = set(binary_results_df.columns) - hard_labels
    exp_to_pred_labels = {
        exp: [
            label
            for label in consider_labels
            if binary_results_df.loc[exp][label] == 1
        ]
        for exp in binary_results_df.index
    }

    print('Computing the most-specific predicted labels...')
    exp_to_ms_pred_labels = {
        exp: label_graph.most_specific_nodes(set(pred_labels) - QUALIFIER_TERMS)
        for exp, pred_labels in exp_to_pred_labels.items()
    }

    # Select cells with highest probability
    exp_to_select_pred_label = {
        exp: max(
            [
                (label, results_df.loc[exp][label])
                for label in ms_pred_labels
            ],
            key=lambda x: x[1]
        )[0]
        for exp, ms_pred_labels in exp_to_ms_pred_labels.items()
        if len(ms_pred_labels) > 0
    } 
   
    # Map each experiment to its finalized label
    exp_to_update_pred = {}
    for exp, select_label in exp_to_select_pred_label.items():
        print('Item {} predicted to be "{} ({})"'.format(
            exp, 
            ou.cell_ontology().id_to_term[select_label].name, 
            select_label
        ))
        all_labels = label_to_ancestors[select_label] 
        exp_to_update_pred[exp] = all_labels

    # Add qualifier cell types
    for exp in exp_to_update_pred:
        for qual_label in QUALIFIER_TERMS:
            if qual_label in exp_to_pred_labels[exp]:
                all_labels = label_to_ancestors[qual_label]
                exp_to_update_pred[exp].update(all_labels)

    # Create dataframe with filtered results
    da = []
    for exp in binary_results_df.index:
        row = []
        for label in binary_results_df.columns:
            if exp in exp_to_update_pred and label in exp_to_update_pred[exp]:
                row.append(1)
            else:
                row.append(0)
        da.append(row)

    df = pd.DataFrame(
        data=da,
        columns=binary_results_df.columns,
        index=binary_results_df.index
    )

    # Most specific cell type labels
    da = []
    for exp in binary_results_df.index:
        if exp in exp_to_select_pred_label:
            da.append(exp_to_select_pred_label[exp])
        else:
            # There was no prediction for this experiment
            da.append('')
    df_ms = pd.DataFrame(
        data=da,
        index = binary_results_df.index,
        columns=['most_specific_cell_type']
    )
    return df, df_ms


def _match_genes(test_genes, all_genes, rsrc_loc, verbose=True, log_dir=None, ret_ids=False):
    # Map each gene to its index
    gene_to_index = {
        gene: index
        for index, gene in enumerate(all_genes)
    }
    if 'ENSG' in test_genes[0] and '.' not in test_genes[0]:
        print("Inferred that input file uses Ensembl gene ID's.")
        if '.' in test_genes[0]:
            print("Inferred that gene ID's have version numbers")
            test_genes = [
                gene_id.split('.')[0]
                for gene_id in test_genes
            ]
        train_genes = sorted(set(test_genes) & set(all_genes))
        not_found = set(all_genes) - set(test_genes)
        train_ids = train_genes
        gene_to_indices = {
            gene: [gene_to_index[gene]]
            for gene in train_genes
        }
    elif 'ENSG' in test_genes[0] and '.' in test_genes[0]:
        print("Inferred that input file uses Ensembl gene ID's with version numbers")
        all_genes = set(all_genes)
        train_ids = []
        train_genes = []
        not_found = []
        gene_to_indices = {}
        for gene in test_genes:
            gene_no_version = gene.split('.')[0]
            if gene_no_version in all_genes:
                train_ids.append(gene_no_version)
                train_genes.append(gene)
                gene_to_indices[gene] = [gene_to_index[gene_no_version]]
            else:
                not_found.append(gene)
    elif len(set(['CD14', 'SOX2', 'NANOG', 'PECAM1']) & set(test_genes)) > 0:
        if verbose:
            print("Inferred that input file uses HGNC gene symbols.")
        genes_f = join(
            rsrc_loc,
            'resources',
            'gene_metadata', 
            'biomart_id_to_symbol.tsv'
        )
        with open(genes_f, 'r') as f:
            sym_to_ids = defaultdict(lambda: [])
            for l in f:
                gene_id, gene_sym = l.split('\t')
                gene_id = gene_id.strip()
                gene_sym = gene_sym.strip()
                sym_to_ids[gene_sym].append(gene_id)
        # Gather training genes
        train_ids = []
        train_genes = []
        all_genes_s = set(all_genes)
        not_found = []
        gene_to_indices = defaultdict(lambda: []) 
        for sym in test_genes:
            if sym in sym_to_ids:
                ids = sym_to_ids[sym]
                for idd in ids:
                    if idd in all_genes_s:
                        train_genes.append(sym)
                        train_ids.append(idd)
                        gene_to_indices[sym].append(gene_to_index[idd])
            else:
                not_found.append(sym)
    else:
        raise ValueError("Unable to determine gene collection. Please make sure the input dataset specifies either HUGO gene symbols or Entrez gene ID's.")
    gene_to_indices = dict(gene_to_indices)
    print('Of {} genes in test set, found {} of {} training set genes in input file.'.format(
        len(test_genes),
        len(train_ids),
        len(all_genes)
    ))
    if log_dir:
        with open(join(log_dir, 'genes_absent_from_training_set.tsv'), 'w') as f:
            f.write('\n'.join(sorted(not_found)))
    return train_genes, gene_to_indices


if __name__ == "__main__":
    main()


