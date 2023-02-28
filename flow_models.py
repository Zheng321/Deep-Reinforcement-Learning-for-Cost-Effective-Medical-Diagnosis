import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(self, layers, leaky=0.0, init_zeros=False, dropout=None):
        """
        :param layers: list of layer sizes from start to end
        :param leaky: slope of the leaky part of the ReLU,
        if 0.0, standard ReLU is used
        :param init_zeros: Flag, if true, weights and biases of last layer
        are initialized with zeros (helpful for deep models, see arXiv 1807.03039)
        :param dropout: Float, if specified, dropout is done before last layer;
        if None, no dropout is done
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers)-2):
            net.append(nn.Linear(layers[k], layers[k+1]))
            # net.append(nn.LeakyReLU(leaky))
            net.append(nn.SiLU())
        if dropout is not None:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-2], layers[-1]))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

    def save_mlp(self, save_path):
        torch.save(self.net.state_dict(), save_path)
        return

    def load_mlp(self, save_path):
        self.net.load_state_dict(torch.load(save_path))
        return
