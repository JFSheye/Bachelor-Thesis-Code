import time
import prepare_dataset as p_d
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegressionCV

# Load data
X_train, X_test, y_train, y_test = p_d.prep_data('mnist', r'F:\Bachelor\Bachelor code\grid\mnist')
#X_train, X_test, y_train, y_test = p_d.prep_data('fashionmnist', r'F:\Bachelor\Bachelor code\grid\fashionmnist')

# Create PCA with 80 components
components = 80
transformer = IncrementalPCA(n_components=components, batch_size=200)

# Transform X_train and X_test
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Instantiate Logistic Regression model
clf = LogisticRegressionCV(cv=5, random_state=1, solver='saga', multi_class='multinomial')

# Do some training
print("\nTraining!\n")

start_time = time.time()
clf.fit(X_train_transformed, y_train)

print(f"Runtime so far: {int(time.time() - start_time)} sec\n")

# Do some testing
score = clf.score(X_test_transformed, y_test)
print(f"The MLP scored {score*100:.2f}% accuracy on the test set")


import pickle
pickle.dump(clf, open("lr.sav", 'wb'))


def n_params(model):
    """Return total number of parameters in a
    Scikit-Learn model.

    This works for the following model types:
     - sklearn.neural_network.MLPClassifier
     - sklearn.neural_network.MLPRegressor
     - sklearn.linear_model.LinearRegression
     - and maybe some others

    Shamelessly found on StackOverflow 
    """
    return (sum([a.size for a in model.coef_]) +
            sum([a.size for a in model.intercept_]))


print(n_params(clf))
