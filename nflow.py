import torch
from torch import nn
import numpy as np

class Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        :param z: input variable, first dimension is batch dim
        :return: transformed z and log of absolute determinant
        """
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        raise NotImplementedError('This flow has no algebraic inverse.')

class Split(Flow):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z1, z2 = z.chunk(2, dim=1)
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        z = torch.cat([z1, z2], 1)
        log_det = 0
        return z, log_det

class Merge(Split):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)

class Permute(Flow):
    def __init__(self, num_channels, mode='shuffle'):
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == 'shuffle':
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(dim=0, index=perm,
                                                       src=torch.arange(self.num_channels))
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

    def forward(self, z):
        if self.mode == 'shuffle':
            z = z[:, self.perm, ...]
        elif self.mode == 'swap':
            z1 = z[:, :self.num_channels // 2, ...]
            z2 = z[:, self.num_channels // 2:, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError('The mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det

    def inverse(self, z):
        if self.mode == 'shuffle':
            z = z[:, self.inv_perm, ...]
        elif self.mode == 'swap':
            z1 = z[:, :(self.num_channels + 1) // 2, ...]
            z2 = z[:, (self.num_channels + 1) // 2:, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError('The mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det

class AffineCoupling(Flow):
    def __init__(self, param_map):
        super().__init__()
        self.param_map = param_map

    def forward(self, z):
        """
        z is a list of z1 and z2; z = [z1, z2]
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1
        """
        z1, z2 = z
        param = self.param_map(z1)
        shift = param[:, 0::2, ...]
        scale = param[:, 1::2, ...]
        if param.shape[-1] % 2 == 1:
            shift = shift[...,1:]
        z2 = z2 * torch.exp(scale) + shift
        log_det = torch.sum(scale, dim=list(range(1, shift.dim()))) # make sure not to take sum in batch_dim

        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)
        shift = param[:, 0::2, ...]
        scale = param[:, 1::2, ...]
        if param.shape[-1] % 2 == 1:
            shift = shift[...,1:]
        z2 = (z2 - shift) * torch.exp(-scale)
        log_det = -torch.sum(scale, dim=list(range(1, shift.dim())))

        return [z1, z2], log_det

class AffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    """
    def __init__(self, param_map):
        super().__init__()
        self.param_map = param_map
        self.flows = nn.ModuleList([])
        self.flows += [Split()]
        self.flows += [AffineCoupling(self.param_map)]
        self.flows += [Merge()]

    def forward(self, z):
        log_det_total = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot

class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """
    def __init__(self, q0, flows, p=None):
        """
        Constructor
        :param q0: Base distribution
        :param flows: List of flows
        :param p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def init_base(self, q):
        self.q0 = q

    def forward(self, z):
        x = z
        for flow in self.flows:
            x, _ = flow(x)
        return x

    def inverse(self, x):
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
        return z

    def sample(self, num_samples=1):
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x):
        """
        Get log probability for batch
        :param x: Batch
        :return: log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        """
        Save state dict of model
        :param path: Path including filename where to save model
        """
        torch.save({'state_dict': self.state_dict(), 'loc':self.q0.loc, 'cov':self.q0.covariance_matrix}, path)

    def load(self, path):
        """
        Load model from state dict
        :param path: Path including filename where to load model from
        """
        d = torch.load(path)
        self.load_state_dict(d['state_dict'])
        self.q0 = torch.distributions.multivariate_normal.MultivariateNormal(d['loc'], covariance_matrix=d['cov'])
