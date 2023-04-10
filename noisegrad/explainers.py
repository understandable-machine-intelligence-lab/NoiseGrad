"""Some examples of explanation methods that can be used in NosieGrad and NoiseGrad++ implementations."""
from __future__ import annotations

from importlib import util

import numpy as np
import torch
import torch.nn as nn

if util.find_spec("captum"):
    from captum.attr import IntegratedGradients, Saliency

    from noisegrad.utils import normalize_heatmap

    def saliency_explainer(
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Implementation of InteGrated Gradients by Sundararajan et al., 2017 using Captum."""

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
            return normalize_heatmap(explanation)

        return explanation

    def intgrad_explainer(
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        abs: bool = True,
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Implementation of InteGrated Gradients by Sundararajan et al., 2017 using Captum."""

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
            explanation = normalize_heatmap(explanation)

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

    def explain_integrated_gradients(
        model: nn.Module,
        input_embeddings: torch.Tensor,
        y_batch: torch.Tensor,
        num_steps: int = 10,
        **kwargs,
    ) -> torch.Tensor:
        def predict_fn(x):
            return model(None, inputs_embeds=x, **kwargs)

        explainer = IntegratedGradients(predict_fn)
        grads = explainer.attribute(
            inputs=input_embeddings, n_steps=num_steps, target=y_batch
        )

        scores = torch.linalg.norm(grads, dim=-1)

        return scores


def explain_gradient_norm(
    model: nn.Module, input_embeddings: torch.Tensor, y_batch: torch.Tensor, **kwargs
) -> torch.Tensor:
    logits = model(None, inputs_embeds=input_embeddings, **kwargs)
    logits_for_class = logits_for_labels(logits, y_batch)
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    scores = torch.linalg.norm(grads, dim=-1)
    return scores


def explain_gradient_x_input(
    model: nn.Module, input_embeddings: torch.Tensor, y_batch: torch.Tensor, **kwargs
) -> torch.Tensor:
    logits = model(None, **kwargs, inputs_embeds=input_embeddings).logits
    logits_for_class = logits_for_labels(logits, y_batch)
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    return torch.sum(grads * input_embeddings, dim=-1).detach()


def logits_for_labels(logits: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
    return logits[torch.arange(0, logits.shape[0], dtype=torch.int), y_batch]
