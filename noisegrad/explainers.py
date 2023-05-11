"""Some examples of explanation methods that can be used in NosieGrad and NoiseGrad++ implementations."""
from __future__ import annotations

from types import SimpleNamespace
import logging

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


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

    def to_rgb(self, score: float, normalize_to_1: bool = False):
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


class image_classification(SimpleNamespace):
    @staticmethod
    def saliency_explainer(
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        from captum.attr import Saliency

        assert (
            len(np.shape(inputs)) == 4
        ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

        explanation = (
            Saliency(model)
            .attribute(inputs, targets, abs=True)
            .sum(axis=1)
            .reshape(
                inputs.shape[0],
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .cpu()
            .data
        )

        if normalize:
            return image_classification.normalize_heatmap(explanation)

        return explanation

    @staticmethod
    def intgrad_explainer(
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        abs: bool = True,
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Implementation of InteGrated Gradients by Sundararajan et al., 2017 using Captum."""

        from captum.attr import IntegratedGradients

        assert (
            len(np.shape(inputs)) == 4
        ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

        explanation = IntegratedGradients(model).attribute(
            inputs=inputs,
            target=targets,
            baselines=torch.zeros_like(inputs, dtype=torch.float32),
        )
        if abs:
            explanation = explanation.abs()
        if normalize:
            explanation = image_classification.normalize_heatmap(explanation)

        return (
            explanation.sum(axis=1)
            .reshape(
                inputs.shape[0],
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .cpu()
            .data
        )

    @staticmethod
    def visualize_explanations(
        image: np.ndarray,
        expl_base: np.ndarray,
        expl_ng: np.ndarray,
        expl_ngp: np.ndarray,
        cmap="gist_gray",
    ):
        import matplotlib.pyplot as plt

        # Plot!
        plt.figure(figsize=(14, 7))

        plt.subplot(1, 4, 1)
        plt.imshow(
            image_classification.denormalize_image(image)
            .transpose(0, 1)
            .transpose(1, 2)
        )
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

    @staticmethod
    def normalize_heatmap(heatmap: np.array) -> np.ndarray:
        """Normalise relevance given a relevance matrix (r) [-1, 1]."""
        if heatmap.min() >= 0.0:
            return heatmap / heatmap.max()
        if heatmap.max() <= 0.0:
            return -heatmap / heatmap.min()
        return (heatmap > 0.0) * heatmap / heatmap.max() - (
            heatmap < 0.0
        ) * heatmap / heatmap.min()

    @staticmethod
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


class text_classification(SimpleNamespace):
    @staticmethod
    def explain_gradient_norm(
        model: nn.Module,
        input_embeddings: torch.Tensor,
        y_batch: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        logits = model(None, inputs_embeds=input_embeddings, **kwargs)
        logits_for_class = text_classification.logits_for_labels(logits, y_batch)
        grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
        scores = torch.linalg.norm(grads, dim=-1)
        return scores

    @staticmethod
    def explain_gradient_x_input(
        model: nn.Module,
        input_embeddings: torch.Tensor,
        y_batch: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        logits = model(None, **kwargs, inputs_embeds=input_embeddings).logits
        logits_for_class = text_classification.logits_for_labels(logits, y_batch)
        grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
        return torch.sum(grads * input_embeddings, dim=-1).detach()

    @staticmethod
    def explain_integrated_gradients(
        model: nn.Module,
        input_embeddings: torch.Tensor,
        y_batch: torch.Tensor,
        num_steps: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        from captum.attr import IntegratedGradients

        def predict_fn(x):
            return model(None, inputs_embeds=x, **kwargs)

        explainer = IntegratedGradients(predict_fn)
        grads = explainer.attribute(
            inputs=input_embeddings, n_steps=num_steps, target=y_batch
        )

        scores = torch.linalg.norm(grads, dim=-1)

        return scores

    @staticmethod
    def visualise_explanations(explanations, labels):
        import matplotlib.pyplot as plt

        h_len = len(explanations)
        v_len = len(explanations[0][0])

        tokens = [i[0] for i in explanations]
        scores = [i[1] for i in explanations]

        fig, axes = plt.subplots(
            h_len,
            v_len,
            figsize=(v_len * 0.75, h_len * 1.25),
            gridspec_kw=dict(left=0.0, right=1.0),
        )
        hspace = 1.0 if labels is not None else 0.1
        plt.subplots_adjust(hspace=hspace, wspace=0.0)
        for i, ax in enumerate(axes):
            color_mapper = ColorMapper(np.max(scores[i]), np.min(scores[i]))
            if labels:
                ax[v_len // 2].set_title(labels[i])
            for j in range(v_len):
                color = color_mapper.to_rgb(scores[i][j], normalize_to_1=True)
                rect = plt.Rectangle((0, 0), 1, 1, color=color)
                ax[j].add_patch(rect)
                ax[j].text(0.5, 0.5, tokens[i][j], ha="center", va="center")
                ax[j].set_xlim(0, 1)
                ax[j].set_ylim(0, 1)
                ax[j].axis("off")
                ax[j] = fig.add_axes([0, 0.05, 1, 0.9], fc=[0, 0, 0, 0])

        ax = axes.ravel()[-1]
        for axis in ["left", "right"]:
            ax.spines[axis].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    @staticmethod
    def logits_for_labels(logits: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        return logits[torch.arange(0, logits.shape[0], dtype=torch.int), y_batch]
