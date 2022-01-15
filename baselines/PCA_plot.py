import numpy as np
import prepare_dataset as p_d
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA

# Load data
# X_train, X_test, y_train, y_test = p_d.prep_data('mnist', r'F:\Bachelor\Bachelor code\grid\mnist')
X_train, X_test, y_train, y_test = p_d.prep_data('fashionmnist', r'F:\Bachelor\Bachelor code\grid\fashionmnist')

# Create PCA with N components
components = 200
transformer = IncrementalPCA(n_components=components, batch_size=200)

# Transform X_train and X_test
X_train_transformed = transformer.fit_transform(X_train)


def create_plot(transformer, components):
    """create plot showing % explained variance"""
    stepsize = 20

    x = [i for i in range(1, components+1)]

    xticks = [i for i in range(0, components+1, stepsize)]
    plt.xticks(xticks)
    plt.xlim(0, x[-1])

    plt.ylim(0, 1)
    plt.yticks([i * 0.1 for i in range(11)])

    plt.xlabel("components")
    plt.ylabel("explained variance")

    plt.grid()

    y = np.cumsum(transformer.explained_variance_ratio_)
    plt.plot(x, y)
    plt.figtext(.5, -.05, f"Explained variance with {components} components: {y[-1] * 100:.2f}%", ha='center')
    plt.title("PCA on fashionMNIST")
    plt.tight_layout()
    plt.savefig("pca_plot_fashionMNIST.png")
    plt.show()


create_plot(transformer=transformer, components=components)
