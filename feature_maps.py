import torch
import numpy as np
import torch.nn.functional as f
from scipy.special import comb


class Sinusoidal:
    """
        Sinusoidal local feature map
        Based on:
            "Supervised Learning With Quantum-Inspired Tensor Networks" (2017) by Stoudenmire and Schwab
    """
    def __init__(self, dims=2):
        self.d = dims

    def execute(self, input_data):
        # TorchMPS tests the validity of a local feature map on a single pixel, which we circumvent here
        n_indices = len(input_data.size())
        if n_indices < 2:
            return torch.rand(self.d)

        # compute the higher dimensional local feature map
        cos = (torch.cos((np.pi / 2.0) * input_data))
        sin = (torch.sin((np.pi / 2.0) * input_data))

        phi = []
        for sj in range(1, (self.d + 1)):
            binomial_coefficient = torch.tensor(comb(self.d - 1, sj - 1))
            square_root = torch.sqrt(binomial_coefficient)

            embedding = square_root * (cos ** (self.d - sj)) * (sin ** (sj - 1))
            phi.append(embedding)
        return torch.cat(phi, dim=-1)


class CNN:
    """
        2-layered Convolutional Neural Network (CNN) local feature map
        Output is normalized, such that each local feature map is a unit vector
    """
    def __init__(self, dims=2):
        super().__init__()

        # GPU usage
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        self.d = dims

        # two layered network
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=self.d//2, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.d//2, out_channels=self.d, kernel_size=(3, 3), padding=1)
        ).to(device)

    def execute(self, img):
        # TorchMPS tests the validity of a local feature map on a single pixel, which we circumvent here
        n_indices = len(img.size())
        if n_indices < 2:
            return torch.rand(self.d)

        # compute the higher dimensional local feature map
        filtered_input = self.network(img)
        imgs = []
        for img in filtered_input:
            flat = [img[i].flatten() for i in range(self.d)]
            imgs.append(torch.stack(flat, dim=-1))

        imgs = torch.stack(imgs)
        return f.normalize(imgs, dim=-1, p=self.d)


class PositionalEncoding:
    """
        Positional Encoding (Scalable Fourier Transforms)
        Based on:
            "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (2020) by Mildenhall et al.
    """
    def __init__(self, dims=2):
        self.d = dims
        self.L = dims // 2

    def execute(self, input_data):
        # TorchMPS tests the validity of a local feature map on a single pixel, which we circumvent here
        n_indices = len(input_data.size())
        if n_indices < 2:
            return torch.rand(self.d)

        # compute the higher dimensional local feature map
        gamma = []
        for Li in range(self.L):
            sin = torch.sin((2 ** Li) * np.pi * input_data)
            cos = torch.cos((2 ** Li) * np.pi * input_data)
            gamma.append(sin)
            gamma.append(cos)

        output = torch.cat(gamma, dim=-1).squeeze(1)
        return output
