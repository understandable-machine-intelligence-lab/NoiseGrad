from __future__ import annotations

import logging
from importlib import util

import numpy as np
import torch

from noisegrad.noisegrad import NoiseType

log = logging.getLogger(__name__)


def apply_noise(
    arr: torch.Tensor, noise: torch.Tensor, noise_type: NoiseType
) -> torch.Tensor:
    if noise_type == "additive":
        return arr + noise
    if noise_type == "multiplicative":
        return arr * noise
    raise ValueError(
        f"Unsupported noise_type, supported are: additive, multiplicative."
    )


def normalize_heatmap(heatmap: np.array) -> np.ndarray:
    """Normalise relevance given a relevance matrix (r) [-1, 1]."""
    if heatmap.min() >= 0.0:
        return heatmap / heatmap.max()
    if heatmap.max() <= 0.0:
        return -heatmap / heatmap.min()
    return (heatmap > 0.0) * heatmap / heatmap.max() - (
        heatmap < 0.0
    ) * heatmap / heatmap.min()


def denormalize_image(
    image,
    mean=torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1),
    std=torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1),
    **params,
):
    """De-normalize a torch image."""
    if isinstance(image, torch.Tensor):
        return (
            image.view(
                [
                    params.get("nr_channels", 3),
                    params.get("img_size", 224),
                    params.get("img_size", 224),
                ]
            )
            * std
        ) + mean
    if isinstance(image, np.ndarray):
        return (image * std.numpy()) + mean.numpy()

    log.error("Make image either a np.array or torch.Tensor before denormalizing.")
    return image


if util.find_spec("matplotlib"):
    import matplotlib
    import matplotlib.pyplot as plt

    def visualize_explanations(
        image: np.ndarray,
        expl_base: np.ndarray,
        expl_ng: np.ndarray,
        expl_ngp: np.ndarray,
        cmap: matplotlib.colors.ListedColormap = "gist_gray",
    ):
        # Plot!
        plt.figure(figsize=(14, 7))

        plt.subplot(1, 4, 1)
        plt.imshow(denormalize_image(image.cpu().data).transpose(0, 1).transpose(1, 2))
        plt.title(f"Original input")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(expl_base, cmap=cmap)
        plt.title(f"Base explanation")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(expl_ng, cmap=cmap)
        plt.title(f"NoiseGrad explanation")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(expl_ngp, cmap=cmap)
        plt.title(f"NoiseGrad++ explanation")
        plt.axis("off")
