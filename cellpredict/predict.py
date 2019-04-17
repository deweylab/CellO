#################################################
#   Run a cell type prediction algorithm
#################################################

import os
from os.path import dirname, join, realpath
import subprocess
from optparse import OptionParser
import pkg_resources as pr
import pandas as pd
from pandas import DataFrame
import cPickle
import onto_lib
from onto_lib import load_ontology

from machine_learning import learners

resource_package = __name__


ALGO_TO_SINGLE_CELL_THRESH = {
    'IR': 0.1,
    'TPR': 0.05,
    'CLR': 0.45
}

ALGO_TO_BULK_THRESH = {
    'IR': 0.15, 
    'TPR': 0.1,
    'CLR': 0.3
}

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-a", "--algo", help="Hierarchical classification algorithm to apply. Must be one of: 'IR' - Isotonic regression, 'CLR' - cascaded logistic regression, 'TPR' - True Path Rule")
    parser.add_option("-s", "--single_cell", action="store_true", help="Perform predictions on single-cell data")
    parser.add_option("-r", "--human_readable_labels", action="store_true", help="Output cell type names rather than Cell Ontology term id's.")
    (options, args) = parser.parse_args()

    queries_f = args[0]
    algo = 'IR'
    if options.algo:
        algo = options.algo
    by_term_name = options.human_readable_labels

    # Set the prediction threshold to achieve 0.9 precision
    if options.single_cell:
        thresh = ALGO_TO_SINGLE_CELL_THRESH[algo]
    else:
        thresh = ALGO_TO_BULK_THRESH[algo]

    # Load the data queries file
    with open(queries_f, 'r') as f:
        queries = [
            [float(x) for x in l.split()]
            for l in f
        ]

    df, df_scores = predict(queries, algo, thresh, by_term_name)
    df.to_csv('predictions.tsv', sep='\t')
    df_scores.to_csv('prediction_scores.tsv', sep='\t')

def predict(queries, algo, thresh, by_term_name=False):

    if by_term_name:
        # TODO 17 is the id for the ontology we want, but
        # this is a bit messy
        og = load_ontology.load('17')[0]
    
    if algo == 'IR':
        model_f = pr.resource_filename(
            resource_package, join("resources", "ir.pickle")
        )
    elif algo == 'CLR':
        model_f = pr.resource_filename(
            resource_package, join("resources", "clr.pickle")
        )
    elif algo == 'TPR':
        model_f = pr.resource_filename(
            resource_package, join("resources", "tpr.pickle")
        )
    with open(model_f, 'r') as f:
        model = cPickle.load(f)
    label_to_confs, label_to_scores = model.predict(queries)
    label_to_confs = [
        {
            list(k)[0]: v
            for k,v in label_to_conf.iteritems()
        }
        for label_to_conf in label_to_confs
    ]

    label_order = label_to_confs[0].keys()
    da = [
        (int(label_to_conf[k] > thresh) for k in label_order)
        for label_to_conf in label_to_confs
    ]
    da_scores = [
        (label_to_conf[k] for k in label_order)
        for label_to_conf in label_to_confs
    ]
    if by_term_name:
        df = DataFrame(
            data=da, 
            columns=[
                og.id_to_term[x].name 
                for x in label_order
            ]
        )
        df_scores = DataFrame(
            data=da_scores, 
            columns=[
                og.id_to_term[x].name 
                for x in label_order
            ]
        )
    else:
        df = DataFrame(data=da, columns=label_order)
        df_scores = DataFrame(data=da_scores, columns=label_order)
    return df, df_scores


if __name__ == "__main__":
    main()


