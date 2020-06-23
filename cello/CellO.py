#######################################################
#   Run CellO
#######################################################

from optparse import OptionParser
from os.path import join
import pandas as pd
from anndata import AnnData
import scanpy as sc
import h5py
import pkg_resources as pr
import dill
from scipy.io import mmread
import numpy as np
import json
from collections import defaultdict

from . import the_ontology
from . import load_expression_matrix
from . import load_training_data
from . import model
from graph_lib.graph import DirectedAcyclicGraph

resource_package = __name__

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
    'IR': 'isotonic_regression'
}
ALGO_TO_PARAMS = {
    'IR': {
        "assert_ambig_neg": False,
        "binary_classifier_algorithm": "logistic_regression",
        "binary_classifier_params": {
            "penalty": "l2",
            "penalty_weight": 1.0,
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

def train_model(ad, algo='IR'):
    genes = ad.var.index

    # Load the training data
    r = load_training_data.load(UNITS)
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
    train_genes, gene_indices = _match_genes(genes, all_genes)

    # Take a subset of the columns for the training-genes 
    X_train = X[:,gene_indices]

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

def predict(
        ad,
        units,
        assay='3_PRIME',
        algo='IR',
        cluster=True,
        res=1.0,
        model_f=None
    ):
    """
    Classify cell types for a given expression matrix.

    Parameters
    ----------
    ad : AnnData object
        Expression matrix of n cells by m genes

    units: String must be one of {'COUNTS', 'CPM', LOG1_CPM, 'TPM', or 'LOG1_TPM'}
        Units of values in ad.X. These keywords refer to the following units:
            COUNTS: raw read counts
            CPM: counts per million
            LOG1_CPM: log(CPM+1)
            TPM: transcripts per million
            LOG1_TPM: log(TPM+1) 

    assay: String, must be one of {'3_PRIME', 'FULL_LENGTH'}
        Whether this matrix is 3-prime-end or full-length sequencing.

    algo: String, optional, default: 'IR'
        The hierarchical classification algorithm to use. Must be one of {'IR', 'CLR'}. 
        These keywords refer to:
            IR: isotonic regression correction
            CLR: cascaded logistic regression

    cluster: Boolean, default: True
        If True, cluster the expression matrix and run classifiation
        on each cluster's mean expression profile

    res: float, default: 1.0
        If cluster=True, then run Leiden with this value for the 
        resolution parameter

    model_f: String, default: None
        A dilled, trained classification model. If None, use a 
        pre-trained default model on all of the genes.

    Returns
    -------
    probabilities, binary_classificaitons : two Pandas DataFrame
        objects, where the first dataframe stores the raw probabilities
        for each cell type across all input cells and the second stores
        the binarized (1 or 0) classifications across all cell types
        for each input cell.
    """

    # Get units into log(TPM+1)
    if assay == FULL_LENGTH_ASSAY:
        if units in set([COUNTS_UNITS, CPM_UNITS, LOG1_CPM_UNITS]):
            print('Error. The input units were specified as {}'.format(units),
                'but the assay was specified as {}.'.format(assay),
                'To run classification, please input expression matrix in ',
                'units of either LOG1_TPM or log(TPM+1) for this assay type.')
            exit() 
    if units == COUNTS_UNITS:
        sc.pp.normalize_total(ad, target_sum=1e6)
        sc.pp.log1p(ad)
    elif units in set([CPM_UNITS, TPM_UNITS]):
        sc.pp.log1p(ad)

    results_df, cell_to_clust = _raw_probabilities(
        ad,
        units,
        assay,
        algo=algo,
        cluster=True,
        res=1.0,
        model_f=model_f
    )

    # Binarize the output probabilities
    threshold_df = _retrieve_empirical_thresholds(ad, algo)
    label_graph = _retrieve_label_graph()
    binary_results_df = _binarize_probabilities(
        results_df,
        threshold_df,
        label_graph
    )

    # Select one most-specific cell type if there are 
    # more than one
    finalized_binary_results_df = _select_one_most_specific(
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
            for cell in sorted(cell_to_clust.keys())
        ]
        results_df = pd.DataFrame(
            data=results_da,
            index=ad.obs.index,
            columns=results_df.columns
        )

        finalized_binary_results_da = [
            finalized_binary_results_df.loc[cell_to_clust[cell]]
            for cell in sorted(cell_to_clust.keys())
        ]
        finalized_binary_results_df = pd.DataFrame(
            data=finalized_binary_results_da,
            index=ad.obs.index,
            columns=finalized_binary_results_df.columns
        )
    return results_df, finalized_binary_results_df


def _raw_probabilities(
        ad, 
        units, 
        assay='3_PRIME', 
        algo='IR', 
        cluster=True, 
        res=1.0,
        model_f=None
    ):
    if model_f is None:
        if algo == 'IR':
            model_f = pr.resource_filename(
                resource_package, 
                join("resources", "trained_models", "ir.dill")
            )
        elif algo == 'CLR':
            model_f = pr.resource_filename(
                resource_package, 
                join("resources", "trained_models", "clr.dill")
            )
    print('Loading model from {}...'.format(model_f))
    with open(model_f, 'rb') as f:
        mod = dill.load(f)
    features = mod.classifier.features

    # Make sure that the genes provided by the user match those expected 
    # by the classifier
    try:
        assert frozenset(features) <= frozenset(ad.var.index)
    except:
        print(features[:20])
        print("Error. The genes present in data matrix do not match those expected by the classifier.")
        print("Please train a classifier on this input gene set using the cello_train_model.py program.")
        exit()

    # Shuffle columns to be in accordance with model
    ad = ad[:,features]

    print(ad.shape)

    # Cluster
    if cluster: 
        cell_to_clust, ad_clust = _cluster(ad, res, units)
        conf_df, score_df = mod.predict(ad_clust.X, ad_clust.obs.index)
    else:
        cell_to_clust = None
        conf_df, score_df = mod.predict(ad.X, ad.obs.index)
    return conf_df, cell_to_clust
 

def _cluster(ad, res, units):
    sc.pp.pca(ad)
    sc.pp.neighbors(ad)
    ad_clust = sc.tl.leiden(ad, resolution=res, copy=True)

    clusters = []
    X_mean_clust = []
    cell_to_clust = {}
    for clust in sorted(set(ad_clust.obs['leiden'])):    
        cells = ad.obs.loc[ad_clust.obs['leiden'] == clust].index
        cell_to_clust.update({
            cell : clust
            for cell in cells
        })
        print('{} cells in cluster {}.'.format(
            len(cells), 
            clust
        ))
        X_clust = ad_clust[cells,:].X
        if units == COUNTS_UNITS:
            x_clust = np.sum(X_clust, axis=0)
            sum_x_clust = float(sum(x_clust))
            x_clust = np.array([x/sum_x_clust for x in x_clust])
            x_clust *= 1e6
            x_clust = np.log(x_clust+1)
            X_mean_clust.append(x_clust)    
        clusters.append(clust)
    X_mean_clust = np.array(X_mean_clust)
    ad_mean_clust = AnnData(
        X=X_mean_clust,
        var=ad.var,
        obs=pd.DataFrame(
            data=clusters,
            index=clusters
        )
    ) 
    return cell_to_clust, ad_mean_clust


def _retrieve_empirical_thresholds(ad, algo):
    # TODO Decided on whether to load the 10x thresholds or all genes thresholds
    if algo == 'IR':
        thresh_f = pr.resource_filename(
            resource_package,
            join("resources", "trained_models", "ir.all_genes_thresholds.tsv")
        )
    elif algo == 'CLR':
        thresh_f = pr.resource_filename(
            resource_package,
            join("resources", "trained_models", "clr.all_genes_thresholds.tsv")
        ) 
    return pd.read_csv(thresh_f, sep='\t', index_col=0)


def _retrieve_label_graph():
    labels_f = pr.resource_filename(
        resource_package,
        join("resources", "training_set", "labels.json")
    )
    with open(labels_f, 'r') as f:
        labels_data = json.load(f)
        source_to_targets= labels_data['label_graph']
        exp_to_labels = labels_data['labels']
    label_graph = DirectedAcyclicGraph(source_to_targets)
    return label_graph


def _binarize_probabilities(
        results_df, 
        decision_df, 
        label_graph
    ):
    # Map each label to its empirical threshold
    label_to_thresh = {
        label: decision_df.loc[label]['empirical_threshold']
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
    og = the_ontology.the_ontology()
    exp_to_update_pred = {}
    for exp, select_label in exp_to_select_pred_label.items():
        print('Item {} predicted to be "{} ({})"'.format(
            exp, 
            og.id_to_term[select_label].name, 
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
            if label in exp_to_update_pred[exp]:
                row.append(1)
            else:
                row.append(0)
        da.append(row)

    df = pd.DataFrame(
        data=da,
        columns=binary_results_df.columns,
        index=binary_results_df.index
    )
    return df

def _match_genes(test_genes, all_genes):
    # Map each gene to its index
    gene_to_index = {
        gene: index
        for index, gene in enumerate(all_genes)
    }
    if 'ENSG' in test_genes[0]:
        print("Inferred that input file uses Ensembl gene Id's.")
        train_genes = sorted(set(test_genes) & set(all_genes))
        gene_indices = [
            gene_to_index[gene]
            for gene in train_genes
        ]
    elif len(set(['CD14', 'SOX2', 'NANOG', 'PECAM1']) & set(test_genes)) > 0:
        print("Inferred that input file uses HGNC gene symbols.")
        genes_f = pr.resource_filename(
            resource_package,
            join("resources", "gene_metadata", "biomart_id_to_symbol.tsv")
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
        for sym in test_genes:
            if sym in sym_to_ids:
                ids = sym_to_ids[sym]
                for idd in ids:
                    if idd in all_genes_s:
                        train_genes.append(sym)
                        train_ids.append(idd)
            else:
                not_found.append(sym)
        gene_indices = [
            gene_to_index[gene]
            for gene in train_ids
        ]
    print('Of {} genes in test set, found {} of {} training set genes in input file.'.format(
        len(test_genes),
        len(train_ids),
        len(all_genes)
    ))
    print('Did not find genes: {}'.format(not_found))
    return train_genes, gene_indices


if __name__ == "__main__":
    main()

