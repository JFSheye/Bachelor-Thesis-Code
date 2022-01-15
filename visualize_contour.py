from utils import create_heatmap


create_heatmap("mnist_linear_1x8_1",
               "Linear feature map (MNIST)",
               fd_min=2, fd_max=2,
               bd_min=4, bd_max=32,
               fd_step_size=1,
               bd_step_size=4)

create_heatmap("mnist_sinusoidal_8x8_1",
               "Sinusoidal feature map (MNIST)",
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4)

create_heatmap("mnist_cnn_8x8_1",
               "Learnable feature map (MNIST)",
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4)

create_heatmap("mnist_positionalEncoding_8x8_1",
               "Positional encoding (MNIST)",
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4)

create_heatmap("fashionmnist_linear_1x8_1",
               "Linear feature map (FashionMNIST)",
               fd_min=2, fd_max=2,
               bd_min=4, bd_max=32,
               fd_step_size=1,
               bd_step_size=4)

create_heatmap("fashionmnist_sinusoidal_8x8_1",
               "Sinusoidal feature map (FashionMNIST)",
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4)

create_heatmap("fashionmnist_cnn_8x8",
               "Learnable feature map (FashionMNIST)",
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4)

create_heatmap("fashionmnist_positionalEncoding_8x8_1",
               "Positional encoding (FashionMNIST)",
               fd_min=4, fd_max=32,
               bd_min=4, bd_max=32,
               fd_step_size=4,
               bd_step_size=4)

'''
# must change create_heatmap() in utils to have this run
create_heatmap("initial_experiment",
               "Sinusoidal feature map (MNIST)",
               fd_min=2, fd_max=32,
               bd_min=2, bd_max=32,
               fd_step_size=2,
               bd_step_size=2)
'''
