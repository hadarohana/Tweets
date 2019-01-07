from sklearn import svm
import sklearn.linear_model.stochastic_gradient as sg
from sklearn.model_selection import GridSearchCV as grid
import numpy
#linear kernel support vector machine using tf-idf vectorizations
class SVM:
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    def __init__(self, train_X, train_Y, test_X, test_Y):
        self.train_X = train_X.apply(lambda x: ' '.join(x)).tolist()
        self.train_Y = train_Y
        self.test_X = test_X.apply(lambda  x: ' '.join(x)).tolist()
        self.test_Y = test_Y
    # Convert text to tf-idf vectors and return accuracy obtained from SVM
    def predict(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        #convert train set to tf-idf vectors
        tf_idf = TfidfVectorizer()
        self.train_X = tf_idf.fit_transform(self.train_X)
        self.test_X = tf_idf.transform( raw_documents=self.test_X)
        #SVM very slow, better suited for task but does not scale to large datasets
        # SVM = svm.SVC(kernel='linear', verbose=True)
        # SVM.fit(X=self.train_X, y=self.train_Y)
        # prediction = SVM.predict(self.test_X)
        # accuracy = numpy.mean(prediction == self.test_Y)
        # param_grid = [
        #     {'alpha': [.00001, .0001, .001, .01]}
        # ] # best results for lowest alpha
        SGD = sg.SGDClassifier(verbose=True, n_iter=3000, alpha=.0000001)
        # clf = grid(SGD, param_grid, cv=3)
        SGD.fit(X=self.train_X, y=self.train_Y)
        prediction = SGD.predict(self.test_X)
        accuracy = numpy.mean(prediction == self.test_Y)

        return accuracy