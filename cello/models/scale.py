
import sklearn
from sklearn import preprocessing

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    vecs = [
        [1.0, 1.0, 3.0, 4.0, 5.0],
        [10.0, 0.0, 5.0, 14.0, 25.0],
        [133.0, 0.0, 35.0, 4.0, 35.0],
        [12.0, 1.0, 32.0, 4.0, 15.0],
    ]

    model = Scale()
    model.fit(vecs)
    print(model.transform(vecs))

class Scale:
    def __init__(self, params):
        self.params = params
        with_std = params['with_std']
        self.model = preprocessing.StandardScaler(
            with_std=with_std
        )

    def fit(self, X):
        print('Fitting scale preprocessor...')
        self.model.fit(X)
        print('done.')

    def transform(self, X):
        print('Scaling data...')
        X_scaled = self.model.transform(X)
        print('done.')
        return X_scaled

if __name__ == "__main__":
    main()
