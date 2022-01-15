import sys
from utils import test_models
from torchvision import transforms, datasets


# === SETUP === #

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Data sets
mnist = datasets.MNIST("./mnist", download=True, transform=transform, train=False)
fashionmnist = datasets.FashionMNIST("./fashionmnist", download=True, transform=transform, train=False)


# === RUN === #
'''
test_models(test_set=fashionmnist,
            fd_min=2, fd_max=2,
            bd_min=4, bd_max=32,
            fd_step_size=1,
            bd_step_size=4,
            flatten_data=True,
            dir_with_models_and_results="fashionmnist_linear_1x8_1")
'''

test_models(test_set=fashionmnist,
            fd_min=4, fd_max=32,
            bd_min=4, bd_max=32,
            fd_step_size=4,
            bd_step_size=4,
            flatten_data=True,
            dir_with_models_and_results="fashionmnist_sinusoidal_8x8_1")
