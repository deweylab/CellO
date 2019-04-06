from optparse import OptionParser

import numpy as np
import json
import random

from classifiers.isotonic_regression import IsotonicRegression
from classifiers.per_label_classifier import PerLabelClassifier
from classifiers.true_path_rule import TruePathRule

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


if __name__ == "__main__":
    main()
