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

class IsotonicRegression(PerLabelClassifier):
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
        super(IsotonicRegression, self).__init__(
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
            #if self.binary_classifier == 'linear_svm':
            #    scores = classifier.decision_function(queries)
            #elif self.binary_classifier == 'l2_logistic_regression':
            scores = [
                x[pos_index] 
                for x in classifier.predict_proba(queries)
            ]
            label_to_scores[label] = scores

        labels_order = sorted(label_to_scores.keys())
        label_to_prob_list = []
        label_to_score_list = []

        constraints_matrix = []
        for row_label in labels_order:
            for constraint_label in self.label_graph.source_to_targets[row_label]:
                row = []
                for label in labels_order:
                    if label == row_label:
                        row.append(1.0)
                    elif label == constraint_label:
                        row.append(-1.0)
                    else:
                        row.append(0.0)
                constraints_matrix.append(row)
        b = np.zeros(len(constraints_matrix))
        constraints_matrix = np.array(constraints_matrix).T

        print "Label order (%d):" % len(labels_order)
        print labels_order
        print "Constraints matrix (%d, %d):" % (len(constraints_matrix), len(constraints_matrix.T))
        print constraints_matrix.T

        for q_i in range(len(queries)):
            Q = np.eye(len(labels_order), len(labels_order))
            predictions = np.array([ # Probabilities
                label_to_scores[label][q_i]
                for label in labels_order
            ])
            predictions = np.array(predictions, dtype=np.double)
            msg_id = random.uniform(0,100) 
            print "%d running solver on item %d/%d..." % (msg_id, q_i+1, len(queries))
            xf, f, xu, iters, lagr, iact = solve_qp(
                Q, 
                predictions, 
                constraints_matrix, 
                b
            )
            print "%d done." % msg_id

            #result = solve_qp_scipy(
            #    Q,
            #    predictions,
            #    constraints_matrix,
            #    b
            #)
            #xf = result.x 
            #print "SECOND GUESSES: %s" % xf

            label_to_prob = {}
            for label, est_prob in zip(labels_order, xf):
                label_to_prob[label] = est_prob
            label_to_prob_list.append(label_to_prob)
            label_to_score = {}
            for label, score in zip(labels_order, predictions):
                label_to_score[label] = label_to_scores[label][q_i]
            label_to_score_list.append(label_to_score)
        return  label_to_prob_list, label_to_score_list


if __name__ == "__main__":
    main()
