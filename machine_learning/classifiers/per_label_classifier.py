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
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import itertools
import math
import random

sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

#import log_p
import onto_lib
from onto_lib import ontology_graph
#import bayesian_network
import binary_classifier as bc

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
    


class PerLabelClassifier(object):
    def __init__(
            self, 
            binary_classifier_type, 
            binary_classifier_params,
            downweight_by_group,
            assert_ambig_neg
        ):
        """
        Args:
            binary_classifier_type: the name of the binary classifier
                to use at each label in the label-graph. Must be
                one of:
                    - linear_svm
                    - l2_logistic_regression
            binary_classifier_params: a dictionary storing the binary
                classifier parameters
            downweight_by_group: if True, then downweight each sample by
                one over the number of samples in its group
        """
        self.binary_classifier_type = str(binary_classifier_type)
        self.binary_classifier_params = binary_classifier_params
        self.label_graph = None

        self.assert_ambig_neg = assert_ambig_neg
        self.downweight_by_group = downweight_by_group

        # Per-label model artifacts
        self.label_to_pos_items = None
        self.label_to_neg_items = None
        self.label_to_classifier = None
        self.label_to_pos_scores = None
        self.label_to_neg_scores = None
        self.label_to_pos_var = None
        self.label_to_neg_var = None
        self.label_to_pos_mean = None
        self.label_to_neg_mean = None

        # all the labels represented in the data
        self.trivial_labels = None

    def fit(
            self, 
            training_feat_vecs, 
            training_items, 
            item_to_labels,  
            label_graph,
            item_to_group=None, 
            verbose=False,
            feat_names=None
        ):
        """
        Args:
            training_feat_vecs: list of training feature vectors 
            training_items: list of item identifiers, each corresponding
                to the feature vector in training_feat_vecs
            item_to_labels: dictionary mapping each item identifier to
                its set of labels in the label_graph
            label_graph: a Graph object representing the label DAG
            item_to_group: a dictionary mapping each item identifier
                to the group it belongs to. These groups may correspond
                to a shared latent variable and should be considered in
                the training process. For example, if the training items
                are gene expression profiles, the groups might be batches.
            verbose: if True, output debugging messages
        """
        self.training_items = training_items 
        self.item_to_labels = item_to_labels
        self.label_graph = label_graph
        self.feat_names = feat_names

        # Map each item to its index in the item lost 
        item_to_index = {x:i for i,x in enumerate(self.training_items)}

        # Map each label to its items
        label_to_items = defaultdict(lambda: set())
        for item in self.training_items:
            labels = self.item_to_labels[item]
            for label in labels:
                label_to_items[label].add(item)
        label_to_items = dict(label_to_items)

        # Compute the positive items for each label
        # This set consists of all items labelled with a 
        # descendent of the label
        print "Computing positive labels..."
        self.label_to_pos_items = {}
        for label in label_to_items:
            positive_items = label_to_items[label].copy()
            desc_labels = self.label_graph.descendent_nodes(label)
            for desc_label in desc_labels:
                if desc_label in label_to_items:
                    positive_items.update(label_to_items[desc_label])
            self.label_to_pos_items[label] = list(positive_items)

        # Compute the negative items for each label
        # This set consists of all items that are not labelled
        # with a descendant of the label, but are also not labelled
        # with an ancestor of the label (can include siblings).
        print "Computing negative labels..."
        self.label_to_neg_items = {}
        for label in label_to_items:
            negative_items = set()
            anc_labels = self.label_graph.ancestor_nodes(label)
            candidate_items = set(self.training_items) - set(self.label_to_pos_items[label])

            if self.assert_ambig_neg:
                self.label_to_neg_items[label] = list(candidate_items)
            else:
                for item in candidate_items:
                    ms_item_labels = self.label_graph.most_specific_nodes(
                        self.item_to_labels[item]
                    )
                    ms_item_labels = set(ms_item_labels)
                    if len(ms_item_labels & anc_labels) == 0:
                        negative_items.add(item)
                self.label_to_neg_items[label] = list(negative_items)


        # TODO REMOVE!!!!!
        for label in self.label_to_pos_items:
            pos_items = set(self.label_to_pos_items[label])
            neg_items = set(self.label_to_neg_items[label])
            all_items = frozenset(pos_items | neg_items)
            print "Are the union of + and - items for label %s all of them?"
            print all_items == frozenset(self.training_items)
        # TODO REMOVE
            

        # Train an SVM classifier at each node of the ontology.
        # We also compute distribution of SVM scores for both 
        # positive and negative classes at each label. The algorithm
        # assumes a normal distribution so we estimate mean and 
        # variance
        self.trivial_labels = set()
        self.label_to_classifier = {}
        self.label_to_pos_var = {}
        self.label_to_neg_var = {}
        self.label_to_pos_mean = {}
        self.label_to_neg_mean = {}
        self.label_to_pos_scores = {}
        self.label_to_neg_scores = {}
        for label_i, curr_label in enumerate(label_to_items.keys()):
            pos_items = self.label_to_pos_items[curr_label]
            neg_items = self.label_to_neg_items[curr_label]

            pos_classes = [1 for x in pos_items]
            neg_classes = [-1 for x in neg_items]
            train_classes = np.asarray(pos_classes + neg_classes)
            train_items = pos_items + neg_items  
 
            pos_feat_vecs = [
                training_feat_vecs[
                    item_to_index[x]
                ]
                for x in pos_items
            ]
            neg_feat_vecs = [
                training_feat_vecs[
                    item_to_index[x]
                ]
                for x in neg_items
            ]
            train_feat_vecs = np.asarray(pos_feat_vecs + neg_feat_vecs) 

            # Train classifier
            verbose = True # TODO REMOVE
            if verbose:
                print "(%d/%d) training classifier for label %s..." % (
                    label_i+1, 
                    len(label_to_items), 
                    ", ".join(curr_label)
                )
                print "Number of positive items: %d" % len(pos_items)
                print "Number of negative items: %d" % len(neg_items)

            if len(pos_items) > 0 and len(neg_items) == 0:
                self.trivial_labels.add(curr_label)
            else:
                model = bc.build_binary_classifier(
                    self.binary_classifier_type, 
                    self.binary_classifier_params
                )

                #if self.downweight_by_group:
                #    assert item_to_group
                #    group_to_num_items = defaultdict(lambda: 0)
                #    for group in item_to_group.values():
                #        group_to_num_items[group] += 1
                #    sample_weights = [
                #        1.0 / group_to_num_items[item_to_group[item]]
                #        for item in train_items
                #    ]
                #    model.fit(
                #        train_items,
                #        train_feat_vecs,
                #        train_classes,
                #        item_to_group,
                #        downweight_by_group=self.downweight_by_group
                #    )
                #else:
                model.fit(
                    train_items,
                    train_feat_vecs,
                    train_classes,
                    item_to_group,
                    downweight_by_group=self.downweight_by_group
                )
                    #model.fit(
                    #    train_feat_vecs,
                    #    train_classes
                    #)
                self.label_to_classifier[curr_label] = model

                # Compute output distributions
                pos_scores = model.decision_function(pos_feat_vecs)
                neg_scores = model.decision_function(neg_feat_vecs)

                self.label_to_pos_scores[curr_label] = pos_scores
                self.label_to_neg_scores[curr_label] = neg_scores
    

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
        label_to_score_list = []
        for q_i in range(len(queries)):
            label_to_score = {}
            for label in label_to_scores.keys():
                label_to_score[label] = label_to_scores[label][q_i]
            label_to_score_list.append(label_to_score)
        return label_to_score_list, label_to_score_list


if __name__ == "__main__":
    main()
