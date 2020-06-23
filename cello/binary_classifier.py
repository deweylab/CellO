import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from sklearn.metrics import average_precision_score
#import random

class L2LogisticRegression():
    def __init__(self, params):
        solver = params['solver']
        penalty_weight = params['penalty_weight']
        if params['downweight_by_class']:
            class_weight = 'balanced'
        else:
            class_weight = None
        if solver == 'liblinear':
            intercept_scaling = params['intercept_scaling']
            self.model = LogisticRegression(
                C=penalty_weight,
                penalty='l2',
                solver='liblinear',
                tol=1e-9,
                class_weight=class_weight,
                intercept_scaling=intercept_scaling
            )

    def fit(self, X, y, sample_weights=None):
        self.model.fit(X, y, sample_weight=sample_weights)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.classes_ = self.model.classes_

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)

    def decision_function(self, X):
        return self.model.decision_function(X)

class SVM():
    def __init__(self, params):
        if params['downweight_by_class']:
            class_weight = 'balanced'
        else:
            class_weight = None
        self.model = SVC(kernel='linear', class_weight=class_weight)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_

    def decision_function(self, X):
        return self.model.decision_function(X)

def build_binary_classifier(algorithm, params):
    if algorithm == "logistic_regression":
        assert 'penalty' in params
        penalty = params['penalty']
        if penalty == 'l2':
            return L2LogisticRegression(params)
    elif algorithm == "svm":
        return SVM(params)
    

if __name__ == "__main__":
    main() 
