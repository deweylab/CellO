################################################################################
#   Hierarchical classification via the True Path Rule algorithm
################################################################################
import pandas as pd
import dill

from .ensemble_binary_classifiers import EnsembleOfBinaryClassifiers 
from graph_lib import graph
   
class TruePathRule():
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

        labels_order = graph.topological_sort(self.label_graph)
        # Re-order columns according to topological order
        confidence_df = confidence_df[labels_order]
        pred_da = []
        for query_i, query in enumerate(X):
            label_to_rec_score = {}
            # Bottom-up pass
            for curr_label in reversed(labels_order):
                children = self.label_graph.source_to_targets[curr_label]
                pos_children = set([
                    label 
                    for label in children
                    if label in confidence_df.columns
                    and confidence_df.iloc[query_i][label] > 0.5
                ])
                pos_children.update([
                    label
                    for label in children
                    if label not in confidence_df.columns
                ])
                if curr_label in confidence_df.columns:
                    curr_label_score = confidence_df.iloc[query_i][curr_label]
                else:
                    curr_label_score = 1.0
                if len(pos_children) > 0:
                    sum_recs_pos_children = sum([
                        label_to_rec_score[label]
                        for label in pos_children
                    ])
                else:
                    sum_recs_pos_children = 0.0
                label_to_rec_score[curr_label] = (1.0 / (1.0 + len(pos_children))) \
                    * (curr_label_score + sum_recs_pos_children) 
                 
            # Top-down pass
            pred_row = [] 
            for curr_label in labels_order:
                parents = self.label_graph.target_to_sources[curr_label]
                if curr_label in confidence_df.columns:
                    curr_label_score = confidence_df.iloc[query_i][curr_label]
                else:
                    curr_label_score = 1.0
                if len(parents) > 0:
                    min_par_rec = min([
                        label_to_rec_score[label]
                        for label in parents
                    ])
                    if min_par_rec < curr_label_score:
                        label_to_rec_score[curr_label] = min_par_rec
                    else:
                        label_to_rec_score[curr_label] = curr_label_score
                else:
                    label_to_rec_score[curr_label] = curr_label_score
       
            pred_da.append([
                label_to_rec_score[label] 
                for label in labels_order
            ])
        pred_df = pd.DataFrame(
            data=pred_da,
            columns=labels_order,
            index=test_items
        )
        return  pred_df, confidence_df


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


if __name__ == "__main__":
    main()
