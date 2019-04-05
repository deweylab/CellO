#################################################################
#   Supervised hierarchical classification using a per-label
#   binary support vector machine. Variants of this algorithm
#   enforce label-graph consistency by propogating positive
#   predictions upward through the graph's 'is_a' relationship
#   edges, and propogates negative predictions downward.
#################################################################

import sys

from optparse import OptionParser
from collections import defaultdict, deque
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from scipy.stats import norm
import itertools
import math
from quadprog import solve_qp
import scipy.optimize
import scipy.stats
import random

sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

#import log_p
import onto_lib
from onto_lib import ontology_graph
from per_label_classifier import PerLabelClassifier
import graph_lib
from graph_lib import graph
#import bayesian_network

DEBUG = False


def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    feat_vecs = [
        [1,1,1,2,3],
        [10,23,1,24,32],
        [543,21,23,2,5]
    ]

    items = [
        'a',
        'b',
        'c'
    ]
    item_to_labels = {
        'a':['hepatocyte', 'disease'],
        'b':['T-cell'],
        'c':['stem cell', 'cultured cell']
    }


   
def solve_qp_scipy(G, a, C, b, meq=0):
    # Minimize     1/2 x^T G x - a^T x
    # Subject to   C.T x >= b
    def f(x):
        return 0.5 * np.dot(x, G).dot(x) - np.dot(a, x)

    if C is not None and b is not None:
        constraints = [{
            'type': 'ineq',
            'fun': lambda x, C=C, b=b, i=i: (np.dot(C.T, x) - b)[i]
        } for i in range(C.shape[1])]
    else:
        constraints = []

    result = scipy.optimize.minimize(f, x0=np.zeros(len(G)), method='COBYLA',
        constraints=constraints, tol=1e-10)
    return result

class TruePathRule(PerLabelClassifier):
    def __init__(
            self, 
            binary_classifier_type, 
            binary_classifier_params, 
            downweight_by_group,
            assert_ambig_neg
        ):
        """
        Args:
            binary_classifier: the name of the binary classifier
                to use at each label in the label-graph.
            binary_classifer_params: a dictionary storing the parameters
                to pass to each binary classifier
            downweight_by_group: if True, then downweight each sample by
                one over the number of samples in its group
            assert_ambig_neg: if True, treat items that are labelled
                most specifically as an ancestral label as negative
        """
        super(TruePathRule, self).__init__(
            binary_classifier_type, 
            binary_classifier_params, 
            downweight_by_group,
            assert_ambig_neg
        )

    def predict(self, queries):
        label_to_scores = {}
        for label, classifier in self.label_to_classifier.iteritems():
            pos_index = 0
            for index, clss in enumerate(classifier.classes_):
                if clss == 1:
                    pos_index = index
                    break
            scores = [
                x[pos_index] 
                for x in classifier.predict_proba(queries)
            ]
            label_to_scores[label] = scores

        labels_order = sorted(label_to_scores.keys())
        label_to_prob_list = []
        label_to_score_list = []

        labels_order = graph.topological_sort(self.label_graph)
        for query_i, query in enumerate(queries):
            label_to_rec_score = {}

            label_to_score = {
                label: label_to_scores[label][query_i]
                for label in label_to_scores
            }
            label_to_score_list.append(label_to_score)

            # Bottom-up pass
            for curr_label in reversed(labels_order):
                children = self.label_graph.source_to_targets[curr_label]
                pos_children = set([
                    label 
                    for label in children
                    if label in label_to_score
                    and label_to_score[label] > 0.5
                ])
                pos_children.update([
                    label
                    for label in children
                    if label not in label_to_score
                ])
                if curr_label in label_to_scores:
                    curr_label_score = label_to_score[curr_label]
                else:
                    curr_label_score = 1.0
                if len(pos_children) > 0:
                    sum_recs_pos_children = sum([
                        label_to_rec_score[label]
                        for label in pos_children
                    ])
                else:
                    sum_recs_pos_children = 0.0
                label_to_rec_score[curr_label] = (1.0 / (1.0 + len(pos_children))) * (curr_label_score + sum_recs_pos_children) 
                 
            # Top-down pass
            for curr_label in labels_order:
                parents = self.label_graph.target_to_sources[curr_label]
                if curr_label in label_to_scores:
                    curr_label_score = label_to_score[curr_label]
                else:
                    curr_label_score = 1.0
                if len(parents) > 0:
                    min_par_rec = min([
                        label_to_rec_score[label]
                        for label in parents
                    ])
                    if min_par_rec < curr_label_score:
                        print "Label %s. Min of parents reconciled scores (MPRS) is %f. Current score is %f. Assigning MPRS" % (curr_label, min_par_rec, curr_label_score)
                        label_to_rec_score[curr_label] = min_par_rec
                    else:
                        print "Label %s. Min of parents reconciled scores (MPRS) is %f. Current score is %f. Assigning current score" % (curr_label, min_par_rec, curr_label_score)
                        label_to_rec_score[curr_label] = curr_label_score
                else:
                    print "Label %s has no parents, so assigning it it's original score of %f" % (curr_label, curr_label_score) 
                    label_to_rec_score[curr_label] = curr_label_score
        
            label_to_prob_list.append(label_to_rec_score)


        return  label_to_prob_list, label_to_score_list


if __name__ == "__main__":
    main()
