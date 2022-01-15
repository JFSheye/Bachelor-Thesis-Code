import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets


# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Data sets
mnist = datasets.MNIST("./mnist", download=True, transform=transform)
fashionmnist = datasets.FashionMNIST("./fashionmnist", download=True, transform=transform)

# Lists of data sets, titles and labels
datasets = [mnist, fashionmnist]
titles = ["mnist", "fashionmnist"]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Visualize data sets
for dataset, title in zip(datasets, titles):
    fig, ax = plt.subplots(2, 5, gridspec_kw=dict(wspace=0.0, hspace=-0.63))

    indices = []
    for label in labels:
        idx = np.where(label == dataset.targets)[0][0]
        indices.append(idx)

    for i in range(2):
        for j in range(5):
            img = dataset.data[indices[i * 5 + j]].numpy()
            ax[i, j].imshow(img, cmap="Greys_r")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    plt.savefig(title + ".png")
    plt.show()

