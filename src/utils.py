import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt


def normalize_heatmap(heatmap: np.array):
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
    elif isinstance(image, np.ndarray):
        std
        return (image * std.numpy()) + mean.numpy()
    else:
        print("Make image either a np.array or torch.Tensor before denormalizing.")
        return image


def visualize_explanations(
    image: torch.Tensor,
    expl_base: torch.Tensor,
    expl_ng: torch.Tensor,
    expl_ngp: torch.Tensor,
    cmap: matplotlib.colors.ListedColormap = "gist_gray",
):
    # Plot!
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 4, 1)
    plt.imshow(denormalize_image(image.cpu().data).transpose(0, 1).transpose(1, 2))
    plt.title(f"Original Input")
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
