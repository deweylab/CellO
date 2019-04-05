from optparse import OptionParser

import numpy as np
import json
import random

from classifiers.one_nn import OneNN
from classifiers.per_label_svm import PerLabelSVM
import classifiers.bayes_net_correction_gibbs
import classifiers.bayes_net_correction
#from classifiers.supervised_probabilistic_nmf import SupervisedMultinomialMatrixFactoriziation
from classifiers.isotonic_regression import IsotonicRegression
from classifiers.per_label_classifier import PerLabelClassifier
#from classifiers.bayes_net_correction_gibbs import PerLabelSVM_BNC_DiscBins_DynamicBins_Gibbs, PerLabelSVM_BNC_DiscBins_StaticBins_Gibbs
from classifiers.bayes_net_correction_gibbs_cv_scores import PerLabelSVM_BNC_DiscBins_DynamicBins_Gibbs, PerLabelSVM_BNC_DiscBins_StaticBins_Gibbs, PerLabelSVM_BNC_DiscBins_NaiveBayes
import classifiers.bayes_net_correction_gibbs_normal_cv_scores
from classifiers.bayes_net_correction_gibbs_normal_cv_scores import BNC_Normal_Gibbs
from classifiers.bayes_net_correction import PerLabelSVM_BNC_DiscBins_DynamicBins
from classifiers.true_path_rule import TruePathRule

#from classifiers.linear_neural_net import LinearNeuralNet 
from classifiers.cascaded_discriminative_classifiers import CascadedDiscriminativeClassifiers_RemoveAmbig, CascadedDiscriminativeClassifiers_AssertAmbigNegative
from classifiers.semi_supervised_cascaded_discriminative_classifiers import SemiSupervisedCascadedDiscriminativeClassifiers 

import dim_reduc.pca
from dim_reduc.pca import PCA
#from dim_reduc.probabilistic_nmf import MultinomialMatrixFactoriziation

import graph_lib
from graph_lib import graph
from graph_lib.graph import DirectedAcyclicGraph


def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    np.random.seed()

    training_feat_vecs = np.random.randn(110, 10)
    training_items = [x for x in range(110)]
    source_to_targets = {
        'A': set(['B', 'J', 'K']),
        'B': set(['C', 'D']),
        'C': set(['E', 'H']),
        'D': set(['E', 'I']),
        'E': set(['F', 'G']),
        'F': set(),
        'G': set(),
        'H': set(['F']),
        'I': set(['G']),
        'J': set(['F']),
        'K': set(['G'])
    }
    label_graph = DirectedAcyclicGraph(source_to_targets)
    item_to_labels = {} 
    for label_i, label in enumerate(graph.topological_sort(label_graph)):
        for i in range(10):
            item = 10*label_i + i
            item_to_labels[item] = label_graph.ancestor_nodes(label)

    item_to_group = {}
    for item in training_items:
        if item % 2 == 0:
            item_to_group[item] = 'even'
        else:
            item_to_group[item] = 'odd'

    print "Item to labels: %s" % item_to_labels
    print "Item to group: %s" % item_to_group

    """    
    model = train_learner(
        'cascaded_discr',
        {},
        training_feat_vecs,
        training_items,
        item_to_labels,
        label_graph,
        dim_reductor_name=None,
        dim_reductor_params=None,
        verbose=False,
        item_to_group=None,
        artifact_dir=None
    )
    """

    gibbs_model = train_learner(
        'svm_bnc_discrete_dynamic_bins_gibbs',
        {
            'kernel': 'linear',
            'pseudocount': 1,
            'prior_positive_prob_estimation': 'from_counts',
            'n_bins_default': 3,
            'n_burn_in': 30000 
        },
        training_feat_vecs,
        training_items,
        item_to_labels,
        label_graph,
        dim_reductor_name=None,
        dim_reductor_params=None,
        verbose=False,
        item_to_group=item_to_group
    )
    junction_tree_model = train_learner(
        'svm_bnc_discrete_dynamic_bins',
        {
            'kernel': 'linear',
            'pseudocount': 1,
            'prior_positive_prob_estimation': 'from_counts',
            'n_bins_default': 3
        },
        training_feat_vecs,
        training_items,
        item_to_labels,
        label_graph,
        dim_reductor_name=None,
        dim_reductor_params=None,
        verbose=False,
        item_to_group=item_to_group,
        artifact_dir="/ua/mnbernstein/Desktop/artifact_dir_test_gibbs"
    )
    test_feat_vecs = np.random.randn(1, 10)
    print "GIBBS PROBABILITIES:"
    print gibbs_model.predict(test_feat_vecs)
    print "JUNCTION TREE PROBABILITIES:"
    print junction_tree_model.predict(test_feat_vecs) 
       
    """ 
    model = train_learner(
        'isotonic_regression',
        {
            'binary_classifier': 'l2_logistic_regression'
        },
        training_feat_vecs,
        training_items,
        item_to_labels,
        label_graph,
        dim_reductor_name=None,
        dim_reductor_params=None,
        verbose=False,
        item_to_group=None,
        artifact_dir=None
    )
    test_feat_vecs = np.random.randn(1, 10)
    print model.predict(test_feat_vecs)
    """

class Model:
    def __init__(self, classifier, dim_reductor=None):
        """
        Args:
            classifier: a classifier object that performs
                supervised classification
            dim_reductor: a dimensonality reduction object
                that performs unsupervised dimensionality
                reduction. If this is supplied, all training
                and classification will be performed on the
                reduced dimensional representation of instances
                as learned by this algorithm.
        """
        self.dim_reductor = dim_reductor
        self.classifier = classifier

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
        if self.dim_reductor:
            self.dim_reductor.fit(training_feat_vecs)
            training_feat_vecs = self.dim_reductor.transform(
                training_feat_vecs
            )
        self.classifier.fit(
            training_feat_vecs,
            training_items,
            item_to_labels,
            label_graph,
            item_to_group=item_to_group,
            verbose=verbose,
            feat_names=feat_names
        )

    def predict(self, queries):
        if self.dim_reductor:
            queries = self.dim_reductor.transform(queries)
        return self.classifier.predict(queries)


def train_learner(
        classifier_name, 
        params, 
        feat_vecs, 
        items, 
        item_to_labels,
        label_graph,
        dim_reductor_name=None,
        dim_reductor_params=None,
        verbose=False,
        item_to_group=None,
        artifact_dir=None,
        feat_names=None
    ):
    """
    Args:
        algorithm: the string representing the machine learning algorithm
        params: a dictioanry storing the parameters for the algorithm
        feat_vecs: the training feature vectors
        items: the list of item identifiers corresponding to each feature
            vector
        item_to_labels: a dictionary mapping each identifier to its labels
        label_graph: a dictionary mapping each label to its neighbors in
            the label-DAG
        verbose: if True, output debugging messages during training and
            predicting
        artifact_dir: if the algorithm requires writing intermediate files
            then the files are placed in this directory
    """

    classifier = None
    if classifier_name == 'knn':
        assert 'metric' in params
        classifier = OneNN(
            metric=params['metric']
        )
    elif classifier_name == 'per_label_svm':
        assert 'kernel' in params
        classifier = PerLabelSVM(
            kernel=params['kernel']
        )
    elif classifier_name == 'per_label_svm_top_down_negative_propagation':
        assert 'kernel' in params
        classifier = PerLabelSVMTopDownNegativePropogation(
            kernel=params['kernel']
        )
    elif classifier_name == 'per_label_svm_bottom_up_positive_propagation':
        assert 'kernel' in params
        classifier = PerLabelSVMBottomUpPositivePropogation(
            kernel=params['kernel']
        )
    elif classifier_name == 'svm_bnc_discrete_dynamic_bins':
        classifier = PerLabelSVM_BNC_DiscBins_DynamicBins(
            kernel=params['kernel'],
            pseudocount=params['pseudocount'],
            prior_pos_estimation=params['prior_positive_prob_estimation'],
            artifact_dir=artifact_dir,
            n_bins_default=params['n_bins_default']
        )
    elif classifier_name == 'bnc_discrete_static_bins_gibbs':
        assert 'n_burn_in' in params
        assert 'binary_classifier_type' in params
        assert 'binary_classifier_type' in params
        assert 'assert_ambig_neg' in params
        assert 'n_bins_score' in params
        assert 'downweight_by_group' in params
        classifier = PerLabelSVM_BNC_DiscBins_StaticBins_Gibbs(
            binary_classifier_type=params['binary_classifier_type'],
            binary_classifier_params=params['binary_classifier_params'],
            assert_ambig_neg=params['assert_ambig_neg'],
            downweight_by_group=params['assert_ambig_neg'],
            pseudocount=params['pseudocount'],
            prior_pos_estimation=params['prior_positive_prob_estimation'],
            n_bins_svm_score=params['n_bins_score'],
            n_burn_in=params['n_burn_in'],
            artifact_dir=artifact_dir,
            constant_prior_pos=params['constant_prior_pos']
        )
    elif classifier_name == 'bnc_discrete_dynamic_bins_gibbs':
        assert 'n_burn_in' in params
        assert 'binary_classifier_type' in params
        assert 'binary_classifier_type' in params
        assert 'assert_ambig_neg' in params
        assert 'downweight_by_group' in params
        classifier = PerLabelSVM_BNC_DiscBins_DynamicBins_Gibbs(
            binary_classifier_type=params['binary_classifier_type'],
            binary_classifier_params=params['binary_classifier_params'],
            assert_ambig_neg=params['assert_ambig_neg'],
            downweight_by_group=params['assert_ambig_neg'],
            pseudocount=params['pseudocount'],
            prior_pos_estimation=params['prior_positive_prob_estimation'],
            n_burn_in=params['n_burn_in'],
            artifact_dir=artifact_dir,
            n_bins_default=params['n_bins_default'],
            constant_prior_pos=params['constant_prior_pos']
        )
    elif classifier_name == 'naive_bayes_dynamic_bins':
        assert 'binary_classifier_type' in params
        assert 'binary_classifier_type' in params
        assert 'assert_ambig_neg' in params
        assert 'downweight_by_group' in params
        classifier = PerLabelSVM_BNC_DiscBins_NaiveBayes(
            binary_classifier_type=params['binary_classifier_type'],
            binary_classifier_params=params['binary_classifier_params'],
            assert_ambig_neg=params['assert_ambig_neg'],
            downweight_by_group=params['assert_ambig_neg'],
            pseudocount=params['pseudocount'],
            artifact_dir=artifact_dir,
            n_bins_default=params['n_bins_default']
        )
    elif classifier_name == 'bnc_normal_gibbs':
        assert 'n_burn_in' in params
        assert 'binary_classifier_type' in params
        assert 'binary_classifier_type' in params
        assert 'assert_ambig_neg' in params
        assert 'downweight_by_group' in params
        assert 'static_std' in params
        classifier = BNC_Normal_Gibbs(
            binary_classifier_type=params['binary_classifier_type'],
            binary_classifier_params=params['binary_classifier_params'],
            assert_ambig_neg=params['assert_ambig_neg'],
            downweight_by_group=params['downweight_by_group'],
            prior_pos_estimation=params['prior_positive_prob_estimation'],
            n_burn_in=params['n_burn_in'],
            artifact_dir=artifact_dir,
            constant_prior_pos=params['constant_prior_pos'],
            static_std=params['static_std']
        )
    elif classifier_name == 'supervised_multinomial_nmf':
        assert 'n_dims' in params
        assert 'n_iters' in params
        assert 'learning_rate' in params
        assert 'topics_alpha' in params
        assert 'topic_props_alpha' in params
        classifier = SupervisedMultinomialMatrixFactoriziation(
            params['n_dims'],
            n_iters=params['n_iters'],
            learning_rate=params['learning_rate'],
            topics_alpha=params['topics_alpha'],
            topic_props_alpha=params['topic_props_alpha']
        )
    elif classifier_name == 'linear_neural_net':
        assert 'n_dims' in params
        assert 'n_iters' in params
        assert 'learning_rate' in params
        classifier = LinearNeuralNet(
            params['n_dims'],
            n_iters=params['n_iters'],
            learning_rate=params['learning_rate']
        )
    elif classifier_name == 'cascaded_discr_assert_ambig_neg':
        assert 'binary_classifier_type' in params
        assert 'binary_classifier_params' in params
        assert 'downweight_by_group' in params
        classifier = CascadedDiscriminativeClassifiers_AssertAmbigNegative(
            binary_classifier_type=params['binary_classifier_type'],
            binary_classifier_params=params['binary_classifier_params'],
            downweight_by_group=params['downweight_by_group']
        )
    elif classifier_name == 'cascaded_discr_remove_ambig':
        assert 'binary_classifier_type' in params
        assert 'binary_classifier_params' in params
        assert 'downweight_by_group' in params
        classifier = CascadedDiscriminativeClassifiers_RemoveAmbig(
            binary_classifier_type=params['binary_classifier_type'],
            binary_classifier_params=params['binary_classifier_params'],
            downweight_by_group=params['downweight_by_group']
        )
    elif classifier_name == 'semi_supervised_cascaded_discr':
        classifier = SemiSupervisedCascadedDiscriminativeClassifiers()
    elif classifier_name == 'isotonic_regression':
        assert 'binary_classifier_type' in params
        assert 'binary_classifier_params' in params
        assert 'assert_ambig_neg' in params
        classifier = IsotonicRegression(
            binary_classifier_type=params['binary_classifier_type'], 
            binary_classifier_params=params['binary_classifier_params'],
            downweight_by_group=params['downweight_by_group'],
            assert_ambig_neg=params['assert_ambig_neg']
        )
    elif classifier_name == 'true_path_rule':
        assert 'binary_classifier_type' in params
        assert 'binary_classifier_params' in params
        assert 'assert_ambig_neg' in params
        classifier = TruePathRule(
            binary_classifier_type=params['binary_classifier_type'],
            binary_classifier_params=params['binary_classifier_params'],
            downweight_by_group=params['downweight_by_group'],
            assert_ambig_neg=params['assert_ambig_neg']
        )
    elif classifier_name == 'per_label_classifier':
        assert 'binary_classifier_type' in params
        assert 'binary_classifier_params' in params
        assert 'assert_ambig_neg' in params
        classifier = PerLabelClassifier(
            binary_classifier_type=params['binary_classifier_type'],
            binary_classifier_params=params['binary_classifier_params'],
            downweight_by_group=params['downweight_by_group'],
            assert_ambig_neg=params['assert_ambig_neg']
        )
    else:
        raise Exception("Algorithm '%s' is not a valid choice" % classifier_name)

    # load the unsupervised dimensionality reduction algorithm
    dim_reductor = None
    if dim_reductor_name:
        assert not dim_reductor_params is None
        assert 'n_dims' in dim_reductor_params
        if dim_reductor_name == 'pca':
            dim_reductor = PCA(
                dim_reductor_params['n_dims']
            )
        if dim_reductor_name == 'multinomial_nmf':
            assert 'n_iters' in dim_reductor_params
            assert 'learning_rate' in dim_reductor_params
            assert 'topics_alpha' in dim_reductor_params
            assert 'topic_props_alpha' in dim_reductor_params
            dim_reductor = MultinomialMatrixFactoriziation(
                dim_reductor_params['n_dims'],
                n_iters=dim_reductor_params['n_iters'],
                learning_rate=dim_reductor_params['learning_rate'],
                topics_alpha = dim_reductor_params['topics_alpha'],
                topic_props_alpha = dim_reductor_params['topic_props_alpha']
            )

    model = Model(
        classifier, 
        dim_reductor=dim_reductor
    )
    model.fit(
        feat_vecs,
        items,
        item_to_labels,
        label_graph,
        item_to_group=item_to_group,
        verbose=verbose,
        feat_names=feat_names
    )
    return model


if __name__ == "__main__":
    main()
