#################################################################
#   Hierarchical classification via isotonic regression
#################################################################
import sys
import numpy as np
from quadprog import solve_qp
import dill
import pandas as pd

from . import model
from .pca import PCA
from .ensemble_binary_classifiers import EnsembleOfBinaryClassifiers 

class IsotonicRegression():
    def __init__(
            self,
            params,
            trained_classifiers_f=None 
        ):
        self.params = params
       
    def fit(
            self,
            X,
            train_items,
            item_to_labels,
            label_graph,
            item_to_group=None,
            verbose=False,
            features=None,
            model_dependency=None
        ):
        """
        model_dependency: String, path to a dilled, pretrained ensemble of binary
            classifiers
        """

        # Either provide the model with a pre-trained ensemble of binary
        # classifiers or train them from scratch
        self.features = features
        if model_dependency is not None:
            with open(model_dependency, 'rb') as f:
                self.ensemble = dill.load(f)
            # Make sure that this pre-trained model was trained on the 
            # same set of items and labels
            assert _validate_pretrained_model(
                self.ensemble, 
                train_items, 
                label_graph,
                features
            )
            self.features = self.ensemble.classifier.features
            self.train_items = self.ensemble.classifier.train_items
            self.label_graph = self.ensemble.classifier.label_graph
        else:
            self.ensemble = EnsembleOfBinaryClassifiers(self.params) 
            self.ensemble.fit(
                X,
                train_items,
                item_to_labels,
                label_graph,
                item_to_group=item_to_group,
                verbose=verbose,
                features=features
            )
            self.features = features
            self.train_items = train_items
            self.label_graph = label_graph

    def predict(self, X, test_items):
        confidence_df, scores_df = self.ensemble.predict(X, test_items)
        labels_order = confidence_df.columns        

        # Create the constraints matrix
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

        pred_da = []
        for q_i in range(len(X)):
            Q = np.eye(len(labels_order), len(labels_order))
            predictions = np.array(
                confidence_df[labels_order].iloc[q_i],
                dtype=np.double
            )
            print("Running solver on item {}/{}...".format(q_i+1, len(X)))
            xf, f, xu, iters, lagr, iact = solve_qp(
                Q, 
                predictions, 
                constraints_matrix, 
                b
            )
            pred_da.append(xf)
        pred_df = pd.DataFrame(
            data=pred_da,
            columns=labels_order,
            index=test_items
        )
        return pred_df, confidence_df

    @property
    def label_to_coefficients(self):
        pca = None
        # This is messy. Basically, this class may be wrapping another model
        # for which PCA is an attribute and we need to extract it
        # TODO Refactor this.
        if isinstance(self.ensemble, model.Model):
            for prep in self.ensemble.preprocessors:
                if isinstance(prep, PCA):
                    pca = prep
            label_to_coefs = self.ensemble.classifier.label_to_coefficients
            if pca is None:
                return label_to_coefs
            else:
                components = pca.components_
                return {
                    label: np.matmul(components.T, coefs.T).squeeze()
                    for label, coefs in label_to_coefs.items()
                }
        else:
            raise Exception('This has not been implemented yet!')
        return {
            label: classif.coef_
            for label, classif in self.ensemble.classifier.label_to_classifier.items()
        }

def _validate_pretrained_model(ensemble, train_items, label_graph, features):
    # Check that the label-graphs have same set of labels
    classif_labels = frozenset(ensemble.classifier.label_graph.get_all_nodes())
    curr_labels = frozenset(label_graph.get_all_nodes())
    if classif_labels != curr_labels:
        return False
    classif_train_items = frozenset(ensemble.classifier.train_items)
    curr_train_items = frozenset(train_items)
    if classif_train_items != curr_train_items:
        return False
    if tuple(ensemble.classifier.features) != tuple(features):
        return False
    return True


