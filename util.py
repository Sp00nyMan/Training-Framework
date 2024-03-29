import numpy as np
import torch
from tqdm import tqdm
from math import log

from torch import Tensor, nn
from torch.utils.data import DataLoader

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device = device
        self.fid = FrechetInceptionDistance(reset_real_features=False, normalize=True).to(device)
        self.kid = KernelInceptionDistance(reset_real_features=False, subset_size=4, normalize=True).to(device)

    def initialize(self, real_data: DataLoader, max_samples=1000):
        for batch, _ in tqdm(real_data, total=max_samples//real_data.batch_size):
            batch = batch.to(self.device)
            if max_samples <= 0:
                break
            batch = batch.detach()
            self.fid.update(batch, True)
            self.kid.update(batch, True)
            max_samples -= batch.shape[0]

    def compute(self, generated_samples: Tensor):
        self.fid.reset()
        self.kid.reset()

        generated_samples = generated_samples.to(self.device)

        self.fid.update(generated_samples, False)
        self.kid.update(generated_samples, False)

        fid = self.fid.compute()
        kid_m, kid_s = self.kid.compute()

        metrics_dict = {
            "FID": fid.item(),
            "KID_mean": kid_m.item(),
            "KID_std": kid_s.item(),

        }
        return metrics_dict


class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.reshape(z.size(0), -1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        return nll


def preprocess_images(images: Tensor, device, n_bits):
    images = images.to(device)

    images *= 255
    if n_bits < 8:
        images = torch.floor(images / 2 ** (8 - n_bits))
    n_bins = 2**n_bits
    images /= n_bins 
    images -= 0.5
    images += torch.rand_like(images) / n_bins

    return images