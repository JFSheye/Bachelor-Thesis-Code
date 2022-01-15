import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from utils import train_model, score
from torchmps import MPS
from feature_maps import *


bond_d = 32

linear_mps = MPS(
    input_dim=2,
    output_dim=2,
    feature_dim=2,
    bond_dim=bond_d,
    adaptive_mode=False,
    periodic_bc=False,
)

feature_d = 14

sinusoidal_mps = MPS(
    input_dim=2,
    output_dim=2,
    feature_dim=feature_d,
    bond_dim=bond_d,
    adaptive_mode=False,
    periodic_bc=False,
)
sinusoidal_mps.register_feature_map(Sinusoidal(dims=feature_d).execute)

positionalEncoding_mps = MPS(
    input_dim=2,
    output_dim=2,
    feature_dim=feature_d,
    bond_dim=bond_d,
    adaptive_mode=False,
    periodic_bc=False,
)
positionalEncoding_mps.register_feature_map(PositionalEncoding(dims=feature_d).execute)

names = [
    "linear",
    "sinusoidal",
    "positional encoding"
]

classifiers = [
    linear_mps,
    sinusoidal_mps,
    positionalEncoding_mps
]


def visualize_toy_data(names, classifiers, runs):
    h = 0.02

    # create data sets
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [
        make_moons(noise=0.1, random_state=0),
        make_circles(noise=0.1, factor=0.5, random_state=1),
        linearly_separable,
    ]

    figure = plt.figure()
    i = 1

    for ds_cnt, ds in enumerate(datasets):
        # preprocess data set, split into training, validation and test part
        X, y = ds

        # scale data to range [0, 1]
        X = ((X - X.min()) / (X.max() - X.min()))

        # scale data to range [-1, 1]
        # X = ((X - X.min()) / (X.max() - X.min())) * (1 - (-1)) + (-1)

        # 80/20 split between X and test set
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 75/25 split between train set and validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
        y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.get_cmap('seismic')
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")

        # Plot the training points and validation points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright, edgecolors="k")

        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
        )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clfscore = 0
            for _ in range(runs):
                clf = train_model(clf, X_train, y_train, X_val, y_val)
                clfscore += score(clf, X_test, y_test)

                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                Z = score(clf, np.c_[xx.ravel(), yy.ravel()], None)

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=1/runs)

            clfscore /= runs

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
            )

            # Plot the validation points
            ax.scatter(
                X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright, edgecolors="k"
            )

            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                edgecolors="k",
                alpha=0.6,
            )

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                xx.max() - 0.1,
                yy.min() + 0.1,
                ("%.2f" % (clfscore)).lstrip("0"),
                size=15,
                horizontalalignment="right",
                color='white'
            )
            i += 1

    plt.tight_layout()
    #plt.savefig("toy_data_B32_d2.png")
    plt.show()


visualize_toy_data(names, classifiers, 5)
