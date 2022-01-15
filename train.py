import sys
import feature_maps as fm
from utils import fit_models
from torchvision import transforms, datasets


# === SETUP === #

# Data transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Data sets
mnist = datasets.MNIST("./mnist", download=True, transform=transform)
fashionmnist = datasets.FashionMNIST("./fashionmnist", download=True, transform=transform)

# Feature maps
linear_map = None
sinusoidal_map = fm.Sinusoidal
cnn_map = fm.CNN
positional_encoding = fm.PositionalEncoding


# === RUN === #

argument = int(sys.argv[1])

if argument == 0:
    fit_models(data_set=mnist,
               feature_map=linear_map,
               fd_min=2, fd_max=2,
               bd_min=4, bd_max=32,
               fd_step_size=1,
               bd_step_size=4,
               flatten_data=True,
               dir_to_save_to="mnist_linear_1x8")

if argument == 1:
    fit_models(data_set=mnist,
               feature_map=sinusoidal_map,
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4,
               flatten_data=True,
               dir_to_save_to="mnist_sinusoidal_8x8")

if argument == 2:
    fit_models(data_set=mnist,
               feature_map=cnn_map,
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4,
               flatten_data=False,
               dir_to_save_to="mnist_cnn_8x8")

if argument == 3:
    fit_models(data_set=mnist,
               feature_map=positional_encoding,
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4,
               flatten_data=True,
               dir_to_save_to="mnist_positionalEncoding_8x8")

if argument == 4:
    fit_models(data_set=fashionmnist,
               feature_map=linear_map,
               fd_min=2, fd_max=2,
               bd_min=4, bd_max=32,
               fd_step_size=1,
               bd_step_size=4,
               flatten_data=True,
               dir_to_save_to="fashionmnist_linear_1x8")

if argument == 5:
    fit_models(data_set=fashionmnist,
               feature_map=sinusoidal_map,
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4,
               flatten_data=True,
               dir_to_save_to="fashionmnist_sinusoidal_8x8")

if argument == 6:
    fit_models(data_set=fashionmnist,
               feature_map=cnn_map,
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4,
               flatten_data=False,
               dir_to_save_to="fashionmnist_cnn_8x8")

if argument == 7:
    fit_models(data_set=fashionmnist,
               feature_map=positional_encoding,
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4,
               flatten_data=True,
               dir_to_save_to="fashionmnist_positionalEncoding_8x8")
