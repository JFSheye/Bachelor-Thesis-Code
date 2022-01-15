import torch
from torchvision import transforms, datasets


def prep_data(dataset, path):
    # Random seed
    torch.manual_seed(1)

    # Data transformations
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.ToTensor()])

    # Load and split data
    if dataset == 'mnist':
        D_train = datasets.MNIST(path, download=True, transform=transform)
        D_test  = datasets.MNIST(path, download=True, transform=transform, train=False)
    elif dataset == 'fashionmnist':
        D_train = datasets.FashionMNIST(path, download=True, transform=transform)
        D_test = datasets.FashionMNIST(path, download=True, transform=transform, train=False)
    else:
        print("Unknown data set")
        return None

    # Converting to numpy arrays somehow changes the pixel values to lie in the
    #  range [0, 255] instead of [0, 1] which is implied in the ToTensor transform
    X_train = D_train.data.numpy() / 255
    X_test = D_test.data.numpy() / 255

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    y_train = D_train.targets.numpy()
    y_test = D_test.targets.numpy()

    return X_train, X_test, y_train, y_test
