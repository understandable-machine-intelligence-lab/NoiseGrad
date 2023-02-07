from __future__ import annotations

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

import logging

log = logging.getLogger(__name__)


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


def normalize_sum_to_1(scores: np.ndarray) -> np.ndarray:
    """Makes the absolute values sum to 1."""
    num_dim = len(scores.shape)
    if num_dim > 2:
        raise ValueError("Only 2D and 1D inputs are supported.")
    if num_dim == 2:
        return np.asarray([normalize_sum_to_1(i) for i in scores])
    scores = scores + np.finfo(np.float32).eps
    return scores / np.abs(scores).sum(-1)


DEFAULT_SPECIAL_TOKENS = [
    "[CLS]",
    "[SEP]",
    "[PAD]",
]


class ColorMapper:

    """
    - Highest score get red (255,0,0).
    - Lowest score gets blue (0,0,255).
    - Positive scores are linearly interpolated between red and white (255, 255, 255).
    - Negative scores are linearly interpolated between blue and white (255, 255, 255).
    """

    def __init__(self, max_score: float, min_score: float):
        self.max_score = max_score
        self.min_score = min_score

    def to_rgb(
        self, score: float, normalize_to_1: bool = False
    ) -> Tuple[float, float, float]:
        k = 1.0 if normalize_to_1 else 255.0

        if score >= 0:
            red = k
            green = k * (1 - score / self.max_score)
            blue = k * (1 - score / self.max_score)
        else:
            red = k * (1 - abs(score / self.min_score))
            green = k * (1 - abs(score / self.min_score))
            blue = k
        return red, green, blue


def _create_div(
    explanation: Tuple[List[str], np.ndarray],
    label: str,
    ignore_special_tokens: bool,
    special_tokens: List[str],
):
    # Create a container, which inherits root styles.
    div_template = """
        <div class="container">
            <p>
                {{label}} <br>
                {{saliency_map}}
            </p>
        </div>
        """

    # For each token, create a separate highlight span with different background color.
    token_span_template = """
        <span class="highlight-container" style="background:{{color}};">
            <span class="highlight"> {{token}} </span>
        </span>
        """
    tokens = explanation[0]
    scores = explanation[1]
    body = ""
    color_mapper = ColorMapper(np.max(scores), np.min(scores))

    for token, score in zip(tokens, scores):
        if ignore_special_tokens and token in special_tokens:
            continue
        red, green, blue = color_mapper.to_rgb(score)
        token_span = token_span_template.replace(
            "{{color}}", f"rgb({red},{green},{blue})"
        )
        token_span = token_span.replace("{{token}}", token)
        body += token_span + " "

    return div_template.replace("{{label}}", label).replace("{{saliency_map}}", body)


def visualise_explanations_as_html(
    explanations: List[Tuple[List[str], np.ndarray]],
    *,
    labels: Optional[List[str]] = None,
    ignore_special_tokens: bool = False,
    special_tokens: List[str] = None,
) -> str:
    """
    Creates a heatmap visualisation from list of explanations.

    Parameters
    ----------
    explanations:
        List of tuples (tokens, salience) containing batch of explanations.
    labels:
        Optional, list of labels to display on top of each explanation.
    ignore_special_tokens:
        If True, special tokens will not be rendered in heatmap.
    special_tokens:
        List of special tokens to ignore during heatmap creation, default= ["[CLS]", "[END]", "[PAD]"].

    Returns
    -------

    html:
        string containing raw html to visualise explanations.

    Examples
    -------

    >>> from IPython.core.display import HTML
    >>> from noisegrad.utils import visualise_explanations_as_html
    >>> a_batch = ...
    >>> raw_html = visualise_explanations_as_html(a_batch)
    >>> HTML(raw_html)

    """
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS

    # Define top-level styles
    heatmap_template = """
        <style>

            .container {
                line-height: 1.4;
                text-align: center;
                margin: 10px 10px 10px 10px;
                color: black;
                background: white;
            }

            p {
                font-size: 16px;
            }

            .highlight-container, .highlight {
                position: relative;
                border-radius: 10% 10% 10% 10%;
            }

            .highlight-container {
                display: inline-block;
            }

            .highlight-container:before {
                content: " ";
                display: block;
                height: 90%;
                width: 100%;
                margin-left: -3px;
                margin-right: -3px;
                position: absolute;
                top: -1px;
                left: -1px;
                padding: 10px 3px 3px 10px;
            }

        </style>

        {{body}}
        """

    spans = ""
    # For each token, create a separate div holding whole input sequence on 1 line.
    for i, explanation in enumerate(explanations):
        label = labels[i] if labels is not None else ""
        div = _create_div(explanation, label, ignore_special_tokens, special_tokens)
        spans += div
    return heatmap_template.replace("{{body}}", spans)
