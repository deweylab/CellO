#################################################################
#   TODO
#################################################################
import sys
from collections import defaultdict
import numpy as np
import math
import pandas as pd

from . import binary_classifier as bc

POS_CLASS = 1
NEG_CLASS = 0

class CascadedDiscriminativeClassifiers(object):
    def __init__(
            self, 
            params
        ):
        """
        Args:
            binary_classifier_type: string, the name of a binary classifier
                to use as the base classifier.
            binary_classifier_params: dictionary mapping the name of a
                parameter to the value for that parameter
        """

        self.binary_classif_algo = params['binary_classifier_algorithm']
        self.binary_classif_params = params['binary_classifier_params']
        self.assert_ambig_neg = params['assert_ambig_neg']

        # Per-label model artifacts
        self.label_to_pos_items = None
        self.label_to_neg_items = None
        self.label_to_classifier = None

        # Labels for which all samples in the training set are
        # annotated with
        self.trivial_labels = None

        # all samples are associated with these labels
        self.trivial_labels = None


    def fit(
            self, 
            X, 
            train_items, 
            item_to_labels,  
            label_graph,
            item_to_group=None, 
            verbose=False,
            features=None,
            model_dependency=None # Unused
        ):
        """
        Args:
            X: NxM matrix of features where N is the number of items and
                M is the number of features
            train_items: list of item identifiers, each corresponding
                to the feature vector in X
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
        self.train_items = train_items 
        self.item_to_labels = item_to_labels
        self.label_graph = label_graph
        self.features = features

        # Map each item to its index in the item lost 
        item_to_index = {
            x:i 
            for i,x in enumerate(self.train_items)
        }

        # Map each label to its items
        self.label_to_items = defaultdict(lambda: set())
        for item in self.train_items:
            labels = self.item_to_labels[item]
            for label in labels:
                self.label_to_items[label].add(item)
        for label in label_graph.get_all_nodes():
            if label not in self.label_to_items:
                self.label_to_items[label] = set()
        self.label_to_items = dict(self.label_to_items)

        # Compute the training sets for each label
        if self.assert_ambig_neg:
            label_to_pos_items, label_to_neg_items = _compute_training_sets_assert_ambiguous_negative(
                self.label_to_items,
                self.item_to_labels,
                self.label_graph
            )
        else:
            label_to_pos_items, label_to_neg_items = _compute_training_sets_remove_ambiguous(
                self.label_to_items,
                self.item_to_labels,
                self.label_graph
            ) 
        self.label_to_pos_items = label_to_pos_items
        self.label_to_neg_items = label_to_neg_items
 
        # Train a classifier at each node of the ontology.
        self.trivial_labels = set()
        self.label_to_classifier = {}
        for label_i, curr_label in enumerate(self.label_to_items.keys()):
            pos_items = self.label_to_pos_items[curr_label]
            neg_items = self.label_to_neg_items[curr_label]
            pos_y = [POS_CLASS for x in pos_items]
            neg_y = [NEG_CLASS for x in neg_items]
            train_items = pos_items + neg_items
            train_y = np.asarray(pos_y + neg_y)
            pos_X = [
                X[
                    item_to_index[x]
                ]
                for x in pos_items
            ]
            neg_X = [
                X[
                    item_to_index[x]
                ]
                for x in neg_items
            ]
            train_X = np.asarray(pos_X + neg_X) 

            # Train classifier
            #if verbose:
            if True:
                print("({}/{}) training classifier for label {}...".format(
                    label_i+1, 
                    len(self.label_to_items), 
                    curr_label
                ))
                print("Number of positive items: {}".format(len(pos_items)))
                print("Number of negative items: {}".format(len(neg_items)))
            if len(pos_items) > 0 and len(neg_items) == 0:
                self.trivial_labels.add(curr_label)
            else:
                model = bc.build_binary_classifier(
                    self.binary_classif_algo,
                    self.binary_classif_params
                )
                model.fit(train_X, train_y)
                self.label_to_classifier[curr_label] = model


    def predict(self, X, test_items):
        # Run all of the classifiers
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
                    for x in classifier.predict_log_proba(X) 
                ]
                label_to_cond_log_probs[label] = probs
            else:
                if len(self.label_to_neg_items[label]) == 0.0:
                    label_to_cond_log_probs[label] = np.zeros(len(X))
                else:
                    raise Exception("No positive items for label %s" % label)

        # Compute the marginals for each label by multiplying all of the conditional
        # probabilities from that label up to the root of the DAG
        label_to_marginals = {}
        for label, log_probs in label_to_cond_log_probs.items():
            products = np.zeros(len(X))
            anc_labels = set(self.label_graph.ancestor_nodes(label)) - set([label])
            for anc_label in anc_labels:
                anc_probs = label_to_cond_log_probs[anc_label]
                products = np.add(products, anc_probs)
            products = np.add(products, log_probs)
            label_to_marginals[label] = np.exp(products)

        # Structure the results into DataFrames
        confidence_df = pd.DataFrame(
            data=label_to_marginals,
            index=test_items
        )        
        scores_df = pd.DataFrame(
            data=label_to_cond_log_probs,
            index=test_items
        )
        sorted_labels = sorted(label_to_marginals.keys())
        confidence_df = confidence_df[sorted_labels]
        scores_df = scores_df[sorted_labels]
        return confidence_df, scores_df


def _compute_training_sets_assert_ambiguous_negative(
        label_to_items,
        item_to_labels,
        label_graph
    ):
    # Compute the positive items for each label
    # This set consists of all items labelled with a 
    # descendent of the label
    print("Computing positive labels...")
    label_to_pos_items = {}
    for curr_label in label_to_items:
        positive_items = label_to_items[curr_label].copy()
        desc_labels = label_graph.descendent_nodes(curr_label)
        for desc_label in desc_labels:
            if desc_label in label_to_items:
                positive_items.update(label_to_items[desc_label])
        label_to_pos_items[curr_label] = list(positive_items)

    # Compute the negative items for each label
    # This set consists of all items that are labelled with the
    # the same parents, but not with the current label
    print("Computing negative labels...")
    label_to_neg_items = {}
    for curr_label in label_to_items:
        negative_items = set()
        parent_labels = set(label_graph.target_to_sources[curr_label])
        for item, labels in item_to_labels.items():
             if frozenset(set(labels) & parent_labels) == frozenset(parent_labels):
                negative_items.add(item)
        negative_items -= set(label_to_pos_items[curr_label])
        label_to_neg_items[curr_label] = list(negative_items)
    return label_to_pos_items, label_to_neg_items
    
        
def _compute_training_sets_remove_ambiguous(
        label_to_items,
        item_to_labels,
        label_graph
    ):
    # Compute the positive items for each label
    # This set consists of all items labelled with a 
    # descendent of the label
    print("Computing positive labels...")
    label_to_pos_items = {}
    for curr_label in label_to_items:
        positive_items = label_to_items[curr_label].copy()
        desc_labels = label_graph.descendent_nodes(curr_label)
        for desc_label in desc_labels:
            if desc_label in label_to_items:
                positive_items.update(label_to_items[desc_label])
        label_to_pos_items[curr_label] = list(positive_items)

    # Compute the negative items for each label
    # This set consists of all items that are labelled with the
    # the same parents, but not with the current label, 
    print("Computing negative labels...")
    label_to_neg_items = {}
    for curr_label in label_to_items:
        negative_items = set()
        parent_labels = set(label_graph.target_to_sources[curr_label])
        for item, labels in item_to_labels.items():
             if frozenset(set(labels) & parent_labels) == frozenset(parent_labels):
                negative_items.add(item)
        negative_items -= set(label_to_pos_items[curr_label])

        # Compute which items are 'ambiguous' for the current label.
        # An item is ambiguous if the parents of the current label
        # are all most-specific labels for the sample
        ambig_items = set()
        for item in negative_items:
            item_ms_labels = label_graph.most_specific_nodes(item_to_labels[item]) 
            if parent_labels <= item_ms_labels:
                ambig_items.add(item)
        negative_items -= ambig_items

        label_to_neg_items[curr_label] = list(negative_items)
    return label_to_pos_items, label_to_neg_items

