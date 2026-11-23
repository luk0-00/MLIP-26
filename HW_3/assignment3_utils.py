"""
Helper code for Assignment III - Generative modeling (SH2150 Machine Learning in Physics).

This file contains code that the notebook needs, but that is not central to the
concepts of the assignment:

    1. The base class for SDEs
    2. Base classes for the noise schedulers alpha_t and beta_t
    3. Probability distributions (Gaussian, Gaussian mixture, image data set)
    4. The UNet scaffolding (the residual layer is implemented in the notebook)
    5. Plotting functions

You do not need to change anything in this file, but feel free to read it.
"""

from abc import ABC, abstractmethod
import math
import os
from typing import List, Optional, Sequence, Tuple, Type

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
import torch
import torch.distributions as D
import torch.nn as nn
from torch.func import vmap, jacrev
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid


# =============================================================================
# 1. SDEs
# =============================================================================

class SDE(ABC):
    """
    Base class for an SDE of the form
        dX_t = u_t(X_t) dt + sigma_t dW_t,
    where u_t is the drift coefficient (vector field) and sigma_t is the diffusion coefficient.
    """

    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient u_t(x_t).
        Args:
            - xt: state at time t, shape (batch_size, ...)
            - t: time, shape (batch_size, 1, ...)
        Returns:
            - drift_coefficient: shape (batch_size, ...)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the diffusion coefficient sigma_t.
        Args:
            - xt: state at time t, shape (batch_size, ...)
            - t: time, shape (batch_size, 1, ...)
        Returns:
            - diffusion_coefficient: shape (batch_size, ...)
        """
        pass


# =============================================================================
# 2. Noise schedulers
# =============================================================================

def _autograd_dt(fn, t: torch.Tensor) -> torch.Tensor:
    """Computes d/dt fn(t) elementwise with automatic differentiation. Output has the same shape as t."""
    dt = vmap(jacrev(fn))(t.unsqueeze(1))
    return dt.view(t.shape)


class Alpha(ABC):
    """
    Base class for the noise scheduler alpha_t. Must satisfy alpha_0 = 0 and alpha_1 = 1.
    If `dt` is not overridden, the time derivative is computed with automatic differentiation.
    """

    def __init__(self):
        assert torch.allclose(self(torch.zeros(1, 1)), torch.zeros(1, 1)), "alpha_0 must be 0"
        assert torch.allclose(self(torch.ones(1, 1)), torch.ones(1, 1)), "alpha_1 must be 1"

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t.
        Args:
            - t: time, shape (num_samples, 1, ...)
        Returns:
            - alpha_t, shape (num_samples, 1, ...)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time, shape (num_samples, 1, ...)
        Returns:
            - d/dt alpha_t, shape (num_samples, 1, ...)
        """
        return _autograd_dt(self, t)


class Beta(ABC):
    """
    Base class for the noise scheduler beta_t. Must satisfy beta_0 = 1 and beta_1 = 0.
    If `dt` is not overridden, the time derivative is computed with automatic differentiation.
    """

    def __init__(self):
        assert torch.allclose(self(torch.zeros(1, 1)), torch.ones(1, 1)), "beta_0 must be 1"
        assert torch.allclose(self(torch.ones(1, 1)), torch.zeros(1, 1), atol=1e-6), "beta_1 must be 0"

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates beta_t.
        Args:
            - t: time, shape (num_samples, 1, ...)
        Returns:
            - beta_t, shape (num_samples, 1, ...)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time, shape (num_samples, 1, ...)
        Returns:
            - d/dt beta_t, shape (num_samples, 1, ...)
        """
        return _autograd_dt(self, t)


# =============================================================================
# 3. Distributions
# =============================================================================

class Sampleable(ABC):
    """Distribution which can be sampled from."""

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (num_samples, ...)
        """
        pass


class Density(ABC):
    """Distribution with a tractable density."""

    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: shape (batch_size, dim)
        Returns:
            - log_density: shape (batch_size, 1)
        """
        pass


class IsotropicGaussian(nn.Module, Sampleable, Density):
    """
    Isotropic Gaussian N(0, std^2 I) producing tensors of arbitrary shape.
    `shape` is the shape of a single sample, e.g. [2] or (C, H, W).
    """

    def __init__(self, shape, std: float = 1.0):
        super().__init__()
        self.shape = shape
        self.std = std
        self.register_buffer("dummy", torch.zeros(1))  # keeps track of the device

    @property
    def dim(self):
        return self.shape

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.std * torch.randn((num_samples, *self.shape), device=self.dummy.device)

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        quad = -(x.pow(2).sum(dim=1)) / (2 * self.std ** 2)
        log_norm_const = -0.5 * self.dim[0] * math.log(2 * math.pi * self.std ** 2)
        return (quad + log_norm_const).unsqueeze(1)


class GaussianMixture(nn.Module, Sampleable, Density):
    """
    Two-dimensional Gaussian mixture model.
    Use the constructors `GaussianMixture.symmetric_2D(...)` or `GaussianMixture.random_2D(...)`.
    """

    def __init__(self, means: torch.Tensor, covs: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            - means: shape (nmodes, 2)
            - covs: shape (nmodes, 2, 2)
            - weights: shape (nmodes,)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> list:
        return [self.means.shape[1]]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means, covariance_matrix=self.covs, validate_args=False,
            ),
            validate_args=False,
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random_2D(cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0, seed: int = 0) -> "GaussianMixture":
        """Mixture of `nmodes` Gaussians with standard deviation `std` placed at random positions."""
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor([1.0, 0.0])
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0) -> "GaussianMixture":
        """Mixture of `nmodes` Gaussians with standard deviation `std` evenly placed on a circle of radius `scale`."""
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale + torch.Tensor([1.0, 0.0]) * x_offset
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)


class ImageSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper around an image data set. Loads all images found (recursively) in
    `<dataset_path>/<split>`, resizes them to `image_size` x `image_size` (if needed) and
    normalizes the pixel values to [-1, 1].

    Args:
        - dataset_path: path to the data set folder, e.g. 'data/brain-stroke'
        - grayscale: True for grayscale images (1 channel), False for RGB images (3 channels).
          By default this is detected from the image files.
        - image_size: images are resized to (image_size, image_size)
        - split: which split of the data set to use ('train', 'val' or 'test')
    """

    def __init__(self, dataset_path: str, grayscale: Optional[bool] = None, image_size: int = 32, split: str = 'train'):
        super().__init__()
        self.dataset_path = dataset_path
        self.grayscale = grayscale
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.dataset = self.load_data()
        self.register_buffer("dummy", torch.zeros(1))  # keeps track of the device

    @property
    def dim(self):
        return self.dataset.shape[1:]

    @property
    def num_channels(self) -> int:
        return self.dataset.shape[1]

    def load_data(self) -> torch.Tensor:
        root = os.path.join(self.dataset_path, self.split)
        if not os.path.isdir(root):
            root = self.dataset_path  # fall back to a flat folder of images
        fpaths = sorted(
            os.path.join(dirpath, fname)
            for dirpath, _, fnames in os.walk(root)
            for fname in fnames
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        if len(fpaths) == 0:
            raise FileNotFoundError(f"No images found in '{root}'. Check `dataset_path`.")

        if self.grayscale is None:
            self.grayscale = Image.open(fpaths[0]).mode in ('L', 'I', 'I;16', 'F')
        mode = 'L' if self.grayscale else 'RGB'
        images = [self.transform(Image.open(f).convert(mode)) for f in tqdm(fpaths, desc='Loading images')]
        return torch.stack(images, dim=0)  # (N, C, H, W)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (num_samples, c, h, w)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds data set size: {len(self.dataset)}")
        indices = torch.randperm(len(self.dataset))[:num_samples]
        return self.dataset[indices].to(self.dummy.device)


# =============================================================================
# 4. UNet scaffolding
# =============================================================================

class FourierEncoder(nn.Module):
    """
    Embeds the time t into a `dim`-dimensional vector using random Fourier features.
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: (bs, 1, 1, 1)
        Returns:
            - embeddings: (bs, dim)
        """
        t = t.view(-1, 1)                          # (bs, 1)
        freqs = t * self.weights * 2 * math.pi     # (bs, half_dim)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1) * math.sqrt(2)  # (bs, dim)


class Encoder(nn.Module):
    """`num_residual_layers` residual layers followed by downsampling by a factor 2."""

    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int,
                 residual_layer: Type[nn.Module]):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            residual_layer(channels_in, t_embed_dim) for _ in range(num_residual_layers)
        ])
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embed)  # (bs, c_in, h, w)
        return self.downsample(x)  # (bs, c_out, h // 2, w // 2)


class Bottleneck(nn.Module):
    """`num_residual_layers` residual layers."""

    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int,
                 residual_layer: Type[nn.Module]):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            residual_layer(channels, t_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, t_embed)  # (bs, c, h, w)
        return x


class Decoder(nn.Module):
    """Upsampling by a factor 2 followed by `num_residual_layers` residual layers."""

    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int,
                 residual_layer: Type[nn.Module]):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
        )
        self.res_blocks = nn.ModuleList([
            residual_layer(channels_out, t_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)  # (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, t_embed)
        return x


class UNet(nn.Module):
    """
    UNet parameterization of a time-dependent vector field (see Figure 1 in the exercise sheet).

    Args:
        - input_channels: number of image channels (1 for grayscale, 3 for RGB)
        - channels: number of channels at each resolution, e.g. [32, 64, 128]
        - num_residual_layers: number of residual layers in each encoder, decoder and the bottleneck
        - t_embed_dim: dimension of the time embedding
        - residual_layer: the residual layer class (implemented in the notebook). Its constructor
          must have the signature residual_layer(channels, t_embed_dim) and its forward method
          the signature forward(x, t_embed).
    """

    def __init__(self, input_channels: int, channels: List[int], num_residual_layers: int, t_embed_dim: int,
                 residual_layer: Type[nn.Module]):
        super().__init__()
        # Initial convolution: (bs, ic, 32, 32) -> (bs, c_0, 32, 32)
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        # Time embedding
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Encoders, bottleneck and decoders
        encoders, decoders = [], []
        for curr_c, next_c in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, residual_layer))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, residual_layer))
        self.encoders = nn.ModuleList(encoders)
        self.bottleneck = Bottleneck(channels[-1], num_residual_layers, t_embed_dim, residual_layer)
        self.decoders = nn.ModuleList(reversed(decoders))

        # Final convolution: (bs, c_0, 32, 32) -> (bs, ic, 32, 32)
        self.final_conv = nn.Conv2d(channels[0], input_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (bs, ic, 32, 32)
            - t: (bs, 1, 1, 1)
        Returns:
            - u_t^theta(x): (bs, ic, 32, 32)
        """
        t_embed = self.time_embedder(t)  # (bs, t_embed_dim)
        x = self.init_conv(x)            # (bs, c_0, 32, 32)

        residuals = []
        for encoder in self.encoders:
            x = encoder(x, t_embed)      # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w // 2)
            residuals.append(x.clone())

        x = self.bottleneck(x, t_embed)

        for decoder in self.decoders:
            x = x + residuals.pop()      # residual connection
            x = decoder(x, t_embed)      # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)

        return self.final_conv(x)        # (bs, ic, 32, 32)


# =============================================================================
# 5. Plotting
# =============================================================================

def _device_of(module) -> torch.device:
    """Returns the device of an nn.Module (cpu if it has no parameters/buffers)."""
    if isinstance(module, nn.Module):
        for tensor in list(module.buffers()) + list(module.parameters()):
            return tensor.device
    return torch.device('cpu')


def _clean_axis(ax: Axes, scale: float):
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def _fmt_t(t) -> str:
    return f'{float(t):.2f}'


def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """Indices of every `record_every`-th time step (always including the last one)."""
    if record_every <= 1:
        return torch.arange(num_timesteps)
    return torch.cat([
        torch.arange(0, num_timesteps - 1, record_every),
        torch.tensor([num_timesteps - 1]),
    ])


def plot_trajectories_1d(trajectories: torch.Tensor, ts: torch.Tensor, ax: Optional[Axes] = None, title: Optional[str] = None):
    """
    Plots trajectories of a one-dimensional SDE.
    Args:
        - trajectories: shape (num_trajectories, num_timesteps, 1), e.g. from `simulator.simulate_with_trajectory`
        - ts: time steps, shape (num_trajectories, num_timesteps, 1)
        - ax: matplotlib Axes to plot on (default: current axes)
        - title: plot title
    """
    if ax is None:
        ax = plt.gca()
    for idx in range(trajectories.shape[0]):
        ax.plot(ts[idx, :, 0].cpu(), trajectories[idx, :, 0].cpu())
    ax.set_xlabel(r'Time ($t$)', fontsize=14)
    ax.set_ylabel(r'$X_t$', fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=14)


def imshow_density(density: Density, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], bins: int = 200,
                   ax: Optional[Axes] = None, **kwargs):
    """Shows the log density of a two-dimensional `density` as an image."""
    if ax is None:
        ax = plt.gca()
    device = _device_of(density)
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins, device=device)
    y = torch.linspace(y_min, y_max, bins, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    with torch.no_grad():
        log_density = density.log_density(xy).reshape(bins, bins).T
    ax.imshow(log_density.cpu(), extent=[x_min, x_max, y_min, y_max], origin='lower', **kwargs)


def _plot_background(ax: Axes, p_noise: Optional[Density], p_data: Optional[Density], scale: float, alpha: float = 0.25):
    """Shows p_noise (grey) and p_data (blue) in the background of a 2D plot."""
    bounds = (-scale, scale)
    if p_noise is not None:
        imshow_density(p_noise, bounds, bounds, ax=ax, vmin=-10, alpha=alpha, cmap=plt.get_cmap('Greys'))
    if p_data is not None:
        imshow_density(p_data, bounds, bounds, ax=ax, vmin=-10, alpha=alpha, cmap=plt.get_cmap('Blues'))


def plot_densities_2d(p_noise: Density, p_data: Density, scale: float = 15.0):
    """Plots the noise distribution, the data distribution, and both together."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['Noise distribution $p_{\\mathrm{noise}}$', 'Toy data distribution $p_{\\mathrm{data}}$',
              '$p_{\\mathrm{noise}}$ and $p_{\\mathrm{data}}$']
    backgrounds = [(p_noise, None), (None, p_data), (p_noise, p_data)]
    for ax, title, (pn, pd) in zip(axes, titles, backgrounds):
        _plot_background(ax, pn, pd, scale, alpha=0.35)
        _clean_axis(ax, scale)
        ax.set_title(title, fontsize=16)
    plt.show()


def hist2d_samples(samples: torch.Tensor, ax: Optional[Axes] = None, bins: int = 100, scale: float = 15.0,
                   percentile: float = 99, **kwargs):
    """Plots a 2D histogram of `samples` (shape (num_samples, 2)) on [-scale, scale]^2."""
    if ax is None:
        ax = plt.gca()
    samples = samples.detach().cpu().numpy()
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=[[-scale, scale], [-scale, scale]])
    norm = Normalize(vmin=0.0, vmax=max(np.percentile(H[H > 0], percentile) if (H > 0).any() else 1.0, 1.0))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    kwargs.setdefault('cmap', 'Blues')
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)


def plot_corruption_2d(xts: Sequence[torch.Tensor], ts: Sequence[float], scale: float = 15.0, kind: str = 'hist',
                       bins: int = 100, title: Optional[str] = None):
    """
    Plots corrupted 2D data points x_t, one panel per time point.
    Args:
        - xts: list of tensors of shape (num_samples, 2), one for each time point in ts
        - ts: list of time points
        - scale: the plots show [-scale, scale]^2
        - kind: 'hist' for 2D histograms or 'scatter' for scatter plots
        - bins: number of histogram bins per axis
        - title: overall figure title
    """
    # constrained layout reserves space for the suptitle above the subplot titles
    fig, axes = plt.subplots(1, len(ts), figsize=(4.5 * len(ts), 5), layout='constrained')
    axes = np.atleast_1d(axes)
    for ax, xt, t in zip(axes, xts, ts):
        if kind == 'hist':
            hist2d_samples(xt, ax=ax, bins=bins, scale=scale)
        elif kind == 'scatter':
            xt = xt.detach().cpu()
            ax.scatter(xt[:, 0], xt[:, 1], s=1, alpha=0.5, color='black')
        else:
            raise ValueError("kind must be 'hist' or 'scatter'")
        _clean_axis(ax, scale)
        ax.set_title(f'$t={_fmt_t(t)}$', fontsize=18)
    if title is not None:
        fig.suptitle(title, fontsize=20)
    plt.show()


def plot_schedulers(schedulers: dict, num_points: int = 200, derivatives: bool = False):
    """
    Plots one or several pairs of noise schedulers.
    Args:
        - schedulers: dictionary {label: (alpha, beta)}
        - derivatives: if True, also plot the time derivatives returned by alpha.dt and beta.dt
    """
    t = torch.linspace(0, 1, num_points).view(-1, 1)
    names = [r'$\alpha_t$', r'$\beta_t$']
    if derivatives:
        names += [r'$\dot\alpha_t$', r'$\dot\beta_t$']
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    for label, (alpha, beta) in schedulers.items():
        curves = [alpha(t), beta(t)]
        if derivatives:
            curves += [alpha.dt(t), beta.dt(t)]  # dt may use autograd, so no torch.no_grad() here
        for ax, curve in zip(axes, curves):
            ax.plot(t, (torch.as_tensor(curve) * torch.ones_like(t)).detach(), label=label)  # also handles constant derivatives
    for ax, name in zip(axes, names):
        ax.set_xlabel('$t$', fontsize=14)
        ax.set_title(name, fontsize=16)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_loss(losses: Sequence[float], title: str = 'Training loss', smoothing: int = 50):
    """Plots the training loss (log scale) together with a moving average."""
    losses = np.asarray(losses)
    plt.figure(figsize=(7, 4))
    plt.plot(losses, alpha=0.4, label='loss')
    if len(losses) > smoothing:
        smoothed = np.convolve(losses, np.ones(smoothing) / smoothing, mode='valid')
        plt.plot(np.arange(smoothing - 1, len(losses)), smoothed, label=f'moving average ({smoothing} epochs)')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_flow_2d(path, xts: torch.Tensor, ts: torch.Tensor, num_marginals: int = 4, scale: float = 15.0,
                 num_trajectories: Optional[int] = None):
    """
    Compares the ground-truth conditional path with samples and trajectories from a learned 2D flow.
    Args:
        - path: the probability path (must have p_noise, p_data, sample_data and sample_xt)
        - xts: simulated trajectories, shape (num_samples, num_timesteps, 2)
        - ts: time steps, shape (num_samples, num_timesteps, 1)
        - num_marginals: number of intermediate time points to show
        - scale: the plots show [-scale, scale]^2
        - num_trajectories: number of trajectories to draw (default: all)
    """
    num_samples, num_timesteps = xts.shape[0], xts.shape[1]
    every_n = record_every(num_timesteps, max(num_timesteps // num_marginals, 1))
    ts_every_n = ts[0, every_n, 0]  # (num_marginals + 1,)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Ground-truth path
    ax = axes[0]
    _plot_background(ax, path.p_noise, path.p_data, scale)
    with torch.no_grad():
        z = path.sample_data(num_samples)
        for t in ts_every_n:
            tt = t.view(1, 1).expand(num_samples, 1)
            x = path.sample_xt(z, tt).cpu()
            ax.scatter(x[:, 0], x[:, 1], marker='o', s=10, alpha=0.5, label=f'$t={_fmt_t(t)}$')
    ax.set_title('Ground-truth conditional path', fontsize=20)
    ax.legend(fontsize=14, loc='upper right', markerscale=2)

    # Samples from the learned flow
    ax = axes[1]
    _plot_background(ax, path.p_noise, path.p_data, scale)
    for i, t in zip(every_n, ts_every_n):
        x = xts[:, i].cpu()
        ax.scatter(x[:, 0], x[:, 1], marker='o', s=10, alpha=0.5, label=f'$t={_fmt_t(t)}$')
    ax.set_title('Samples from the learned flow', fontsize=20)
    ax.legend(fontsize=14, loc='upper right', markerscale=2)

    # Trajectories of the learned flow
    ax = axes[2]
    _plot_background(ax, path.p_noise, path.p_data, scale)
    n_traj = num_samples if num_trajectories is None else min(num_trajectories, num_samples)
    for idx in range(n_traj):
        ax.plot(xts[idx, :, 0].cpu(), xts[idx, :, 1].cpu(), alpha=0.3, color='black', lw=0.8)
    ax.set_title('Trajectories of the learned flow', fontsize=20)

    for ax in axes:
        _clean_axis(ax, scale)
    plt.tight_layout()
    plt.show()


def _image_grid(x: torch.Tensor, nrow: int) -> torch.Tensor:
    grid = make_grid(x.detach().cpu(), nrow=nrow, normalize=True, value_range=(-1, 1))
    return grid.permute(1, 2, 0)  # (H, W, C)


def show_images(x: torch.Tensor, nrow: int = 3, title: Optional[str] = None, ax: Optional[Axes] = None):
    """
    Shows a batch of images in a grid.
    Args:
        - x: images with pixel values in [-1, 1], shape (num_images, c, h, w)
        - nrow: number of images per row
        - title: plot title
        - ax: matplotlib Axes to plot on (default: a new figure)
    """
    if ax is None:
        plt.figure(figsize=(1.5 * nrow + 1, 1.5 * math.ceil(len(x) / nrow) + 1))
        ax = plt.gca()
    ax.imshow(_image_grid(x, nrow), interpolation='none')
    ax.axis('off')
    if title is not None:
        ax.set_title(title, fontsize=14)


def plot_corruption_images(xts: Sequence[torch.Tensor], ts: Sequence[float], nrow: int = 3):
    """
    Plots corrupted images x_t, one image grid per time point.
    Args:
        - xts: list of tensors of shape (num_images, c, h, w), one for each time point in ts
        - ts: list of time points
        - nrow: number of images per row in each grid
    """
    fig, axes = plt.subplots(1, len(ts), figsize=(4 * len(ts), 4.5))
    axes = np.atleast_1d(axes)
    for ax, xt, t in zip(axes, xts, ts):
        show_images(xt, nrow=nrow, title=f'$t={_fmt_t(t)}$', ax=ax)
    plt.tight_layout()
    plt.show()
