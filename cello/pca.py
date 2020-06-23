from optparse import OptionParser
import sklearn
from sklearn import decomposition
import dill

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    vecs = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 2.0, 5.0, 14.0, 25.0],
        [133.0, 2.0, 35.0, 4.0, 35.0],
        [12.0, 2.0, 32.0, 4.0, 15.0],
    ]
    model = PCA(2)
    model.fit(vecs)
    print(model.transform(vecs))

class PCA:
    def __init__(self, params):
        n_dims = params['n_components']
        self.n_dims = n_dims
        self.model = decomposition.PCA(n_dims, svd_solver='randomized')

    def fit(self, X):
        print('Fitting PCA with {} components...'.format(self.n_dims))
        self.model.fit(X)
        print('done.')

    @property
    def components_(self):
        return self.model.components_

    def transform(self, X):
        print('Transforming with PCA...')
        X = self.model.transform(X)
        print('done.')
        return X

if __name__ == "__main__":
    main()
