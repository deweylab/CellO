##################################################################################################
#                      Runs the Kallisto program.
##################################################################################################

import os
from os.path import dirname, join, realpath
import subprocess
from optparse import OptionParser
import pkg_resources as pr
import pandas as pd
from pandas import DataFrame
import cPickle

from machine_learning import learners

resource_package = __name__

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-a", "--algo", help="Hierarchical classification algorithm to apply. Must be one of: 'IR' - Isotonic regression, 'CLR' - cascaded logistic regression, 'TPR' - True Path Rule")
    parser.add_option("-s", "--single_cell_precision", help="Output hard predictions for a given estimated precision-scores according to performance on a single-cell test set")
    parser.add_option("-b", "--bulk_precision", help="Output hard predictions for a given estimated precision-scores according to performance on a bulk RNA-seq test set")
    (options, args) = parser.parse_args()

    queries_f = args[0]
    algo = 'IR'
    if options.algo:
        algo = options.algo

    with open(queries_f, 'r') as f:
        queries = [
            [float(x) for x in l.split()]
            for l in f
        ]

    label_to_confs = predict(queries, algo)
    label_order = label_to_confs[0].keys()

    da = [
        (label_to_conf[k] for k in label_order)
        for label_to_conf in label_to_confs
    ]
    df = DataFrame(data=da, columns=label_order)
    df.to_csv('prediction_results.tsv', sep='\t')

def predict(queries, algo):
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
    return label_to_confs 


if __name__ == "__main__":
    main()


