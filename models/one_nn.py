#################################################################
#   One-nearest neighbor classifier
#################################################################

from optparse import OptionParser
import sklearn
from sklearn.neighbors import NearestNeighbors
import scipy 
from scipy.stats import entropy
import numpy as np
import math

from . import model_utils

def main():
    a = [0.3, 0.2, 0.5]
    b = [0.4, 0.1, 0.5]

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
    model = OneNN('correlation')
    model.fit(feat_vecs, items, item_to_labels)
    print(model.predict([[10,10,10,20,30]]))


def jensen_shannon(a, b):
    m = [
        0.5 * (a[i] + b[i])
        for i in range(len(a))
    ]
    return math.sqrt(0.5 * (entropy(a, m) + entropy(b, m)))


class OneNN:
    def __init__(self, params):
        metric = params['metric']
        if metric == 'correlation':
            self.metric_func = scipy.spatial.distance.correlation
        elif metric == 'euclidean':
            self.metric_func = scipy.spatial.distance.euclidean
        elif metric == 'jensen_shannon':
            self.metric_func = jensen_shannon
        self.metric = metric
        self.model = None
        self.item_to_labels = None

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
        self.items = train_items
        self.item_to_labels = item_to_labels
        self.label_graph = label_graph
        self.model = NearestNeighbors(
            metric=self.metric_func
        )
        self.model.fit(X)
        self.training_labels = set()
        for labels in self.item_to_labels.values():
            self.training_labels.update(labels)
        self.features = features

    def predict(self, X, test_items):
        print('Finding nearest neighbors for {} samples...'.format(len(X)))
        neighb_dists, neighb_sets = self.model.kneighbors(
            X, 
            n_neighbors=1, 
            return_distance=True
        )
        print('done.')
        dists =[x[0] for x in neighb_dists]
        neighbs = [x[0] for x in neighb_sets]
        pred_labels = []

        label_to_conf_list = []
        for dist, neighb in zip(dists, neighbs):
            label_to_conf = {}
            item = self.items[neighb]
            neighb_labels = self.item_to_labels[item]
            for label in self.training_labels:
                if label in neighb_labels:
                    label_to_conf[label] = -1.0 * dist
                else:
                    label_to_conf[label] = float('-inf')
            label_to_conf_list.append(
                label_to_conf
            )
        conf_matrix = model_utils.convert_to_matrix(
            label_to_conf_list,
            test_items
        )
        return conf_matrix, conf_matrix 

            
if __name__ == "__main__":
    main()
