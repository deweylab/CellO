#################################################################
#   TODO
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

sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

#import log_p
import onto_lib
from onto_lib import ontology_graph
import binary_classifier as bc

POS_CLASS = 1
NEG_CLASS = 0

DEBUG = True


class CascadedDiscriminativeClassifiers(object):
    def __init__(self, binary_classifier_type, binary_classifier_params, downweight_by_group):
        """
        Args:
            binary_classifier_type: string, the name of a binary classifier
                to use as the base classifier.
            binary_classifier_params: dictionary mapping the name of a
                parameter to the value for that parameter
        """
        # per-label model artifacts
        self.label_to_pos_items = None
        self.label_to_neg_items = None
        self.label_to_classifier = None
        self.downweight_by_group = downweight_by_group

        # all samples are associated with these labels
        self.trivial_labels = None

        # base classifier
        self.binary_classifier_type = binary_classifier_type
        self.binary_classifier_params = binary_classifier_params

    def fit(
            self, 
            training_feat_vecs, 
            training_items, 
            item_to_labels,  
            label_graph,
            regularizer_C=1.0,
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
        item_to_index = {
            x:i 
            for i,x in enumerate(self.training_items)
        }

        # Map each label to its items
        self.label_to_items = defaultdict(lambda: set())
        for item in self.training_items:
            labels = self.item_to_labels[item]
            for label in labels:
                self.label_to_items[label].add(item)
        for label in label_graph.get_all_nodes():
            if label not in self.label_to_items:
                self.label_to_items[label] = set()
        self.label_to_items = dict(self.label_to_items)

        # Compute the training sets for each label
        self._compute_training_sets()
        
        # Train a classifier at each node of the ontology.
        self.trivial_labels = set()
        self.label_to_classifier = {}
        for label_i, curr_label in enumerate(self.label_to_items.keys()):
            pos_items = self.label_to_pos_items[curr_label]
            neg_items = self.label_to_neg_items[curr_label]
            pos_classes = [POS_CLASS for x in pos_items]
            neg_classes = [NEG_CLASS for x in neg_items]
            train_items = pos_items + neg_items
            train_classes = np.asarray(pos_classes + neg_classes)
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
            #if verbose:
            if True:
                print "(%d/%d) training classifier for label %s..." % (
                    label_i+1, 
                    len(self.label_to_items), 
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
                mod_item_to_group = {
                    item: item_to_group[item]
                    for item in train_items
                }
                model.fit(
                    train_items,
                    train_feat_vecs, 
                    train_classes,
                    mod_item_to_group,
                    downweight_by_group=self.downweight_by_group
                )
                self.label_to_classifier[curr_label] = model

    def predict(self, queries):
        label_to_cond_log_probs = {}
        for label in self.label_graph.get_all_nodes():
            if label in self.label_to_classifier:
                classifier = self.label_to_classifier[label]
                pos_indices = [
                    i 
                    for i, classs in enumerate(classifier.classes_) 
                    if classs == POS_CLASS
                ]
                assert len(pos_indices) == 1
                pos_index = pos_indices[0]
                probs = [
                    x[pos_index] 
                    for x in classifier.predict_log_proba(queries) 
                ]
                label_to_cond_log_probs[label] = probs
            else:
                if len(self.label_to_neg_items[label]) == 0.0:
                    label_to_cond_log_probs[label] = np.zeros(len(queries))
                else:
                    raise Exception("No positive items for label %s" % label)
        # Compute the marginals for each label by multiplying all of the conditional
        # probabilities from that label up to the root of the DAG
        label_to_marginals = {}
        for label, log_probs in label_to_cond_log_probs.iteritems():
            products = np.zeros(len(queries))
            anc_labels = set(self.label_graph.ancestor_nodes(label)) - set([label])
            for anc_label in anc_labels:
                anc_probs = label_to_cond_log_probs[anc_label]
                products = np.add(products, anc_probs)
            products = np.add(products, log_probs)
            label_to_marginals[label] = products

        # Gather the results
        label_to_marginal_list = []
        label_to_cond_prob_list = []
        for query_i in range(len(queries)):
            label_to_marginal = {}
            for label, marginals in label_to_marginals.iteritems():
                label_to_marginal[label] = math.exp(marginals[query_i])
            label_to_cond_prob = {}
            for label, log_probs in label_to_cond_log_probs.iteritems():
                label_to_cond_prob[label] = math.exp(log_probs[query_i])
            label_to_marginal_list.append(label_to_marginal)
            label_to_cond_prob_list.append(label_to_cond_prob)
        return label_to_marginal_list, label_to_cond_prob_list
   

class CascadedDiscriminativeClassifiers_AssertAmbigNegative(CascadedDiscriminativeClassifiers):
    def __init__(
            self,  
            binary_classifier_type, 
            binary_classifier_params,
            downweight_by_group
        ):
        super(CascadedDiscriminativeClassifiers_AssertAmbigNegative, self).__init__( 
            binary_classifier_type, 
            binary_classifier_params,
            downweight_by_group
        ) 

    def _compute_training_sets(self):
        # Compute the positive items for each label
        # This set consists of all items labelled with a 
        # descendent of the label
        print "Computing positive labels..."
        self.label_to_pos_items = {}
        for curr_label in self.label_to_items:
            positive_items = self.label_to_items[curr_label].copy()
            desc_labels = self.label_graph.descendent_nodes(curr_label)
            for desc_label in desc_labels:
                if desc_label in self.label_to_items:
                    positive_items.update(self.label_to_items[desc_label])
            self.label_to_pos_items[curr_label] = list(positive_items)
            print "Label %s, positive items (%d): %s" % (curr_label, len(self.label_to_pos_items[curr_label]), self.label_to_pos_items[curr_label])

        # Compute the negative items for each label
        # This set consists of all items that are labelled with the
        # the same parents, but not with the current label
        print "Computing negative labels..."
        self.label_to_neg_items = {}
        for curr_label in self.label_to_items:
            negative_items = set()
            parent_labels = set(self.label_graph.target_to_sources[curr_label])
            print "Parent labels of %s are %s" % (curr_label, parent_labels) # TODO REMOVE
            for item, labels in self.item_to_labels.iteritems():
                 if frozenset(labels & parent_labels) == frozenset(parent_labels):
                    negative_items.add(item)
            negative_items -= set(self.label_to_pos_items[curr_label])
            self.label_to_neg_items[curr_label] = list(negative_items)
            print "Label %s, negative items (%d): %s" % (curr_label, len(self.label_to_neg_items[curr_label]), self.label_to_neg_items[curr_label])

            
class CascadedDiscriminativeClassifiers_RemoveAmbig(CascadedDiscriminativeClassifiers):
    def __init__(
            self, 
            binary_classifier_type, 
            binary_classifier_params,
            downweight_by_group
        ):
        super(CascadedDiscriminativeClassifiers_RemoveAmbig, self).__init__( 
            binary_classifier_type, 
            binary_classifier_params,
            downweight_by_group
        )

    def _compute_training_sets(self):
        # Compute the positive items for each label
        # This set consists of all items labelled with a 
        # descendent of the label
        print "Computing positive labels..."
        self.label_to_pos_items = {}
        for curr_label in self.label_to_items:
            positive_items = self.label_to_items[curr_label].copy()
            desc_labels = self.label_graph.descendent_nodes(curr_label)
            for desc_label in desc_labels:
                if desc_label in self.label_to_items:
                    positive_items.update(self.label_to_items[desc_label])
            self.label_to_pos_items[curr_label] = list(positive_items)
            print "Label %s, positive items (%d): %s" % (curr_label, len(self.label_to_pos_items[curr_label]), self.label_to_pos_items[curr_label])

        # Compute the negative items for each label
        # This set consists of all items that are labelled with the
        # the same parents, but not with the current label, 
        print "Computing negative labels..."
        self.label_to_neg_items = {}
        for curr_label in self.label_to_items:
            negative_items = set()
            parent_labels = set(self.label_graph.target_to_sources[curr_label])
            print "Parent labels of %s are %s" % (curr_label, parent_labels) # TODO REMOVE
            for item, labels in self.item_to_labels.iteritems():
                 if frozenset(labels & parent_labels) == frozenset(parent_labels):
                    negative_items.add(item)
            negative_items -= set(self.label_to_pos_items[curr_label])

            # Compute which items are 'ambiguous' for the current label.
            # An item is ambiguous if the parents of the current label
            # are all most-specific labels for the sample
            ambig_items = set()
            for item in negative_items:
                item_ms_labels = self.label_graph.most_specific_nodes(self.item_to_labels[item]) 
                if parent_labels <= item_ms_labels:
                    ambig_items.add(item)
            negative_items -= ambig_items

            self.label_to_neg_items[curr_label] = list(negative_items)
            print "Label %s, negative items (%d): %s" % (curr_label, len(self.label_to_neg_items[curr_label]), self.label_to_neg_items[curr_label])
            print "Ambiguous items (%d): %s" % (len(ambig_items), ambig_items)


if __name__ == "__main__":
    main()
