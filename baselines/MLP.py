import time
import prepare_dataset as p_d
import pickle
from sklearn.neural_network import MLPClassifier


# Load data
X_train, X_test, y_train, y_test = p_d.prep_data('mnist', r'F:\Bachelor\Bachelor code\grid\mnist')
#X_train, X_test, y_train, y_test = p_d.prep_data('fashionmnist', r'F:\Bachelor\Bachelor code\grid\fashionmnist')

# Instantiate MLP model
clf = MLPClassifier(random_state=1,
                    hidden_layer_sizes=(100),
                    early_stopping=True,
                    validation_fraction=0.166,
                    batch_size=100)

# Do some training
print("\nTraining!\n")

start_time = time.time()
clf.fit(X_train, y_train)

print(f"Runtime so far: {int(time.time() - start_time)} sec\n")

pickle.dump(clf, open("mlp.sav", 'wb'))

# Do some testing
score = clf.score(X_test, y_test)
print(f"The MLP scored {score*100:.2f}% accuracy on the test set")
