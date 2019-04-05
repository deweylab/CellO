import sklearn
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
import random
from collections import defaultdict
import itertools

def main():
    group_to_items = {
        'a': [1,2],
        'b': [3,4,5],
        'c': [6,7,8,9],
        'd': [10],
        'e': [11, 12, 13]
    }
    item_to_class = {
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 1,
        11: 1,
        12: 1,
        13: 0
    }
    print _partition_groups(group_to_items, item_to_class)

DEFAULT_C = 1.0

def _partition_groups(group_to_items, item_to_class):
    """
    Partition the groups into two groups in an attempt
    to ensure that the partitions are even-sized. This
    is the knapsack problem, and we solve it here using
    hill-climbing.
    """
    def _obj_fun(
            group_to_incl,
            group_to_items
        ):
        """
        n_total = sum([
            len(items)
            for items in group_to_items.values()
        ])
        n_included = sum([
            len(items)
            for group, items  in group_to_items.iteritems()
            if group_to_incl[group]
        ])
        frac = float(n_included) / n_total
        diff_half = abs(frac - 0.5)
        """

        max_class_weight = 1
        min_class_weight = 5 

        all_classes = list(set(item_to_class.values()))
        n_total_class_0 = len([
            item
            for item, cls in item_to_class.iteritems()
            if cls == all_classes[0]
        ])
        n_included_class_0 = sum([
            len([
                item
                for item in group_to_items[group]
                if item_to_class[item] == all_classes[0]
            ])
            for group, incl in group_to_incl.iteritems()
            if incl
        ])

        n_total_class_1 = len([
            item
            for item, cls in item_to_class.iteritems()
            if cls == all_classes[1]
        ])
        n_included_class_1 = sum([
            len([
                item 
                for item in group_to_items[group]
                if item_to_class[item] == all_classes[1]
            ])
            for group, incl in group_to_incl.iteritems()
            if incl
        ])

        #print "Current inclusion: %s" % group_to_incl

        frac_class_0 = float(n_included_class_0) / n_total_class_0
        frac_class_1 = float(n_included_class_1) / n_total_class_1
        #print "Fraction included in class 0: %d/%d=%f" % (n_included_class_0, n_total_class_0, frac_class_0)
        #print "Fraction included in class 1: %d/%d=%f" % (n_included_class_1, n_total_class_1, frac_class_1)
        diff_half_class_0 = abs(frac_class_0 - 0.5)
        diff_half_class_1 = abs(frac_class_1 - 0.5)        
        
        if n_total_class_0 > n_total_class_1:
            class_0_diff_weight = max_class_weight
            class_1_diff_weight = min_class_weight
        else:
            class_0_diff_weight = min_class_weight
            class_1_diff_weight = max_class_weight

        obj_func_val = (class_0_diff_weight * diff_half_class_0) + (class_1_diff_weight * diff_half_class_1)
        #print "Objective function = (W_0=%f)*(diff_half_0=%f)+(W_1=%f)*(diff_half_1=%f) = %f" % (class_0_diff_weight, diff_half_class_0, class_1_diff_weight, diff_half_class_1, obj_func_val)
        return obj_func_val

    def _hill_climb(
            group_to_incl,
            group_to_items
        ):
        curr_obj = _obj_fun(
            group_to_incl,
            group_to_items
        )
        obj_improved = True
        while obj_improved:
            obj_improved = False
            for group in group_to_incl:
                group_to_incl[group] = not group_to_incl[group]
                obj = _obj_fun(
                    group_to_incl,
                    group_to_items
                )
                if obj < curr_obj:
                    obj_improved = True
                    curr_obj = obj
                else:
                    group_to_incl[group] = not group_to_incl[group]
        return curr_obj, group_to_incl
    
    groups = set(group_to_items.keys())
    min_group_to_included = None
    min_obj = float('inf')

    iters = 10
    #random.seed(88)
    for iter_i in range(iters):
        # Randomly assign each group to be included
        # or not to be included
        group_to_incl = {
            group: bool(random.randint(0, 1))
            for group in groups
        }
        obj, group_to_incl = _hill_climb(
            group_to_incl,
            group_to_items
        )
        if obj < min_obj:
            min_group_to_included = {
                k:v
                for k,v in group_to_incl.iteritems()
            }
            min_obj = obj
    return min_group_to_included


class BinaryClassifier(object):
    def __init__(self, params):
        self.params = params

    def _cross_validate_params(
            self,
            items,
            vecs,
            the_classes,
            item_to_group,
            cand_params,
            build_model_func,
            default_param,
            downweight_by_group=False
        ):
        item_to_index = {
            item: index
            for index, item in enumerate(items)
        }

        classes_tuple = tuple(set(the_classes))

        # map each group to the items it contains
        group_to_items = defaultdict(lambda: set())
        for item, group in item_to_group.iteritems():
            group_to_items[group].add(item)
        group_to_items = dict(group_to_items)

        # map each group to the classes in that group
        group_to_classes = defaultdict(lambda: set())
        for item, clas in zip(items, the_classes):
            group_to_classes[item_to_group[item]].add(clas)
        group_to_classes = dict(group_to_classes)

        pure_class_0_groups = set([
            group
            for group, classes in group_to_classes.iteritems()
            if len(classes) == 1
            and list(classes)[0] == classes_tuple[0]
        ])
        pure_class_1_groups = set([
            group
            for group, classes in group_to_classes.iteritems()
            if len(classes) == 1
            and list(classes)[0] == classes_tuple[1]
        ])
        mixed_class_groups = set([
            group
            for group, classes in group_to_classes.iteritems()
            if len(classes) == 2
        ])

        n_groups_class_0 = len(pure_class_0_groups) + len(mixed_class_groups)
        n_groups_class_1 = len(pure_class_1_groups) + len(mixed_class_groups)

        print "no. groups of class 0: %d" % n_groups_class_0
        print "no. groups of class 1: %d" % n_groups_class_1
        if n_groups_class_0 < 2 or n_groups_class_1 < 2:
            print "One of the classes is represented in only one group. Thus, cannot do 2-fold CV. Returning default parameters=%s." % str(default_param)
            return default_param

        item_to_class = {
            item: cls
            for item, cls in zip(items, the_classes)
        }

        group_to_incl = _partition_groups(group_to_items, item_to_class)
        part_1_groups = set([
            group
            for group, incl in group_to_incl.iteritems()
            if incl
        ])
        part_2_groups = set([
            group
            for group, incl in group_to_incl.iteritems()
            if not incl
        ])

        max_avg_prec = -1.0
        max_param = None
        for curr_params in cand_params:
            print "performing cross-validation on params=%s..." % str(curr_params)
            # Create the training and test sets for each fold
            print "partitioning data into folds..."
            train_1_vecs = []
            train_1_classes = []
            train_1_weights = []
            train_2_vecs = []
            train_2_classes = []
            train_2_weights = []
            for group in part_1_groups:
                for item in group_to_items[group]:
                    train_1_vecs.append(vecs[item_to_index[item]])
                    train_1_classes.append(the_classes[item_to_index[item]])
                    train_1_weights.append(1.0 / len(group_to_items[group]))
                    test_2_vecs = train_1_vecs
                    test_2_classes = train_1_classes
            for group in part_2_groups:
                for item in group_to_items[group]:
                    train_2_vecs.append(vecs[item_to_index[item]])
                    train_2_classes.append(the_classes[item_to_index[item]])
                    train_2_weights.append(1.0 / len(group_to_items[group]))
                    test_1_vecs = train_2_vecs
                    test_1_classes = train_2_classes

            # Run the first fold
            model_1 = build_model_func(curr_params, downweight_by_group)
            print "running first fold..."
            if downweight_by_group:
                model_1.fit(train_1_vecs, train_1_classes, sample_weight=train_1_weights)
            else:
                model_1.fit(train_1_vecs, train_1_classes)
            pos_index = 0
            for index, clss in enumerate(model_1.classes_):
                if clss == 1:
                    pos_index = index
                    break
            scores_1 = [
                x[pos_index]
                for x in model_1.predict_proba(test_1_vecs)
            ]

            # Run the second fold
            print "running second fold..."
            model_2 = build_model_func(curr_params, downweight_by_group)
            if downweight_by_group:
                model_2.fit(train_2_vecs, train_2_classes, train_2_weights)
            else:
                model_2.fit(train_2_vecs, train_2_classes)
            pos_index = 0
            for index, clss in enumerate(model_2.classes_):
                if clss == 1:
                    pos_index = index
                    break
            scores_2 = [
                x[pos_index]
                for x in model_2.predict_proba(test_2_vecs)
            ]

            avg_prec = average_precision_score(
                test_1_classes + test_2_classes,
                scores_1 + scores_2
            )
            if avg_prec > max_avg_prec:
                max_avg_prec = avg_prec
                max_param = curr_params
        print "cross-fold validation yielded parameters:\n%s" % str(max_param)
        return max_param

    def fit(
            self,
            the_items,
            the_vecs,
            the_classes,
            item_to_group,
            downweight_by_group=False
        ):
        raise Exception("Can't call fit on the abstract class BinaryClassifier")

    def predict(self, queries):
        return self.model.predict_proba(queries)

    def predict_proba(self, queries):
        return self.model.predict_proba(queries)

    def predict_log_proba(self, queries):
        return self.model.predict_log_proba(queries)

    def decision_function(self, queries):
        return self.model.decision_function(queries) 

class ElasticNetLogisticRegression(BinaryClassifier):
    def __init__(self, params):
        super(ElasticNetLogisticRegression, self).__init__(
            params
        )

    def _get_model(self, penalty_weights, downweight_by_group=False):
        """
        Args:
            penalty_weights: first element is the penalty weight for
                the l1 penalty, the second is for the l2 penalty
        """
        assert len(penalty_weights) == 2 
        l1_weight = penalty_weights[0]
        l2_weight = penalty_weights[1]
        alpha = l1_weight + l2_weight
        l1_ratio = l1_weight / (l1_weight + l2_weight)
        model = SGDClassifier(
            loss='log', 
            penalty='elasticnet',
            max_iter = 1000,
            alpha=alpha,
            l1_ratio=l1_ratio
        )
        return model

    def fit(
            self,
            the_items,
            the_vecs,
            the_classes,
            item_to_group,
            downweight_by_group=False
        ):
        item_to_group = {
            item:group
            for item, group in item_to_group.iteritems()
            if item in set(the_items)
        }
        if self.params['penalty_weight'] == "cross_validate":
            cand_cs = [0.01, 0.1, 1.0, 10, 100]
            cand_c_pairs = [
                weight_pair
                for weight_pair in itertools.product(cand_cs, cand_cs)
            ]
            chosen_c_pair = self._cross_validate_params(
                the_items,
                the_vecs,
                the_classes,
                item_to_group,
                cand_c_pairs,
                self._get_model,
                (DEFAULT_C, DEFAULT_C),
                downweight_by_group=False
            )
        elif self.params['penalty_weight'] == "specified":
            assert "l2_weight" in self.params
            assert "l1_weight" in self.params
            chosen_c_pair = (
                self.params["l1_weight"],
                self.params["l2_weight"]
            )
        self.model = self._get_model(chosen_c_pair)
        # map each group to the items it contains
        group_to_items = defaultdict(lambda: set())
        for item, group in item_to_group.iteritems():
            group_to_items[group].add(item)
        group_to_n_items = {
            group: float(len(items))
            for group, items in group_to_items.iteritems()
        }
        sample_weights = [
            1.0 / group_to_n_items[item_to_group[item]]
            for item in the_items
        ]
        if downweight_by_group:
            self.model.fit(
                the_vecs,
                the_classes,
                sample_weight=sample_weights
            )
        else:
            self.model.fit(
                the_vecs,
                the_classes
            )
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.classes_ = self.model.classes_
 

class L1LogisticRegression(BinaryClassifier):
    def __init__(self, params):
        super(L1LogisticRegression, self).__init__(
            params
        )

    def _get_model(self, penalty_weight, downweight_by_group=False):
        assert not downweight_by_group
        model = LogisticRegression(
            C=penalty_weight,
            penalty='l1',
            solver='liblinear',
            tol=1e-9
        )
        return model
    
    def fit(
            self,
            the_items,
            the_vecs,
            the_classes,
            item_to_group,
            downweight_by_group=False
        ):
        item_to_group = {
            item:group
            for item, group in item_to_group.iteritems()
            if item in set(the_items)
        }
        if 'penalty_weight' not in self.params:
            chosen_c = DEFAULT_C
        elif self.params['penalty_weight'] == 'cross_validate':
            cand_cs = [0.01, 0.1, 1.0, 10, 100]
            chosen_c = self._cross_validate_params(
                the_items,
                the_vecs,
                the_classes,
                item_to_group,
                cand_cs,
                self._get_model,
                DEFAULT_C,
                downweight_by_group=False
            )
        else:
            chosen_c = self.params['penalty_weight']
        self.model = self._get_model(chosen_c)
        #self.model.fit(
        #    the_vecs,
        #    the_classes
        #)
        #self.coef_ = self.model.coef_
        #self.intercept_ = self.model.intercept_
        #self.classes_ = self.model.classes_

        # map each group to the items it contains
        group_to_items = defaultdict(lambda: set())
        for item, group in item_to_group.iteritems():
            group_to_items[group].add(item)
        group_to_n_items = {
            group: float(len(items))
            for group, items in group_to_items.iteritems()
        }
        sample_weights = [
            1.0 / group_to_n_items[item_to_group[item]]
            for item in the_items
        ]
        if downweight_by_group:
            self.model.fit(
                the_vecs,
                the_classes,
                sample_weight=sample_weights
            )
        else:
            self.model.fit(
                the_vecs,
                the_classes
            )
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.classes_ = self.model.classes_   
 

class L2LogisticRegression(BinaryClassifier):
    def __init__(self, params):
        super(L2LogisticRegression, self).__init__(
            params
        )

    def _get_model(self, penalty_weight, downweight_by_group=False):
        if downweight_by_group:
            # we're going to need to use the sample_weight
            # argument, which is only supported by the 
            # newton-cg solver
            model = LogisticRegression(
                C=penalty_weight,
                penalty='l2',
                solver='newton-cg',
                tol=1e-9
            )
            return model
        else:
            # since we don't need the sample_weight
            # argument, we prefer the liblinear solver
            # because it is faster
            model = LogisticRegression(
                C=penalty_weight,
                penalty='l2',
                solver='liblinear',
                tol=1e-9
            )
            return model
        
    def fit(
            self,
            the_items,
            the_vecs,
            the_classes,
            item_to_group,
            downweight_by_group=False
        ):
        item_to_group = {
            item:group
            for item, group in item_to_group.iteritems()
            if item in set(the_items)
        }
        if 'penalty_weight' not in self.params:
            chosen_c = DEFAULT_C
        elif self.params['penalty_weight'] == 'cross_validate':
            cand_cs = [0.01, 0.1, 1.0, 10, 100]
            chosen_c = self._cross_validate_params(
                the_items,
                the_vecs,
                the_classes,
                item_to_group,
                cand_cs,
                self._get_model,
                DEFAULT_C,
                downweight_by_group=False
            )
        else:
            chosen_c = self.params['penalty_weight']
        self.model = self._get_model(
            chosen_c,
            downweight_by_group
        )

        # map each group to the items it contains
        group_to_items = defaultdict(lambda: set())
        for item, group in item_to_group.iteritems():
            group_to_items[group].add(item)
        group_to_n_items = {
            group: float(len(items))
            for group, items in group_to_items.iteritems()
        }
        sample_weights = [
            1.0 / group_to_n_items[item_to_group[item]]
            for item in the_items
        ]
        if downweight_by_group:
            self.model.fit(
                the_vecs,
                the_classes,
                sample_weight=sample_weights
            )
        else:
            self.model.fit(
                the_vecs,
                the_classes
            )
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.classes_ = self.model.classes_


class LinearSVM(BinaryClassifier):
    def __init__(self, params):
        super(LinearSVM, self).__init__(
            params
        )

    def fit(
            self,
            the_items,
            the_vecs,
            the_classes,
            item_to_group,
            downweight_by_group=False
        ):
        self.model = SVC(kernel='linear', probability=True, tol=1e-6)

        # map each group to the items it contains
        group_to_items = defaultdict(lambda: set())
        for item, group in item_to_group.iteritems():
            group_to_items[group].add(item)
        group_to_n_items = {
            group: float(len(items))
            for group, items in group_to_items.iteritems()
        }
        sample_weights = [
            1.0 / group_to_n_items[item_to_group[item]]
            for item in the_items
        ]
        if downweight_by_group:
            self.model.fit(
                the_vecs,
                the_classes,
                sample_weight=sample_weights
            )
        else:
            self.model.fit(
                the_vecs,
                the_classes
            )
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.classes_ = self.model.classes_

def build_binary_classifier(classif_name, classif_params):
    if classif_name == "logistic_regression":
        assert 'penalty' in classif_params
        penalty = classif_params['penalty']
        if penalty == 'l2':
            return L2LogisticRegression(classif_params)
        elif penalty == 'l1':
            return L1LogisticRegression(classif_params)   
        elif penalty == 'elastic_net':
            return ElasticNetLogisticRegression(classif_params) 
    elif classif_name == "svm":
        assert 'kernel' in classif_params
        if classif_params['kernel'] == 'linear':
            return LinearSVM(classif_params)
        #kernel = classif_params['kernel'].encode('utf-8')
        #return SVC(kernel=kernel, probability=True)
    
    """
    if classif_name == "logistic_regression":
        assert 'C' in classif_params
        if 'penalty' in classif_params and classif_params['penalty'] == 'l1':
            return LogisticRegression(
                C=classif_params['C'],
                penalty=classif_params['penalty']
            )
        else:
            return LogisticRegression(
                C=classif_params['C'],
                solver='newton-cg'
            )
    """ 

if __name__ == "__main__":
    main() 
