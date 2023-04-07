"""Some examples of explanation methods that can be used in NosieGrad and NoiseGrad++ implementations."""
from __future__ import annotations

import torch
import torch.nn as nn
from importlib import util

from typing import Optional
import numpy as np

if util.find_spec("captum"):
    from noisegrad.utils import normalize_heatmap
    from captum.attr import Saliency, IntegratedGradients

    def saliency_explainer(
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        normalize: bool = False,
        **kwargs,
    ) -> np.array:
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
    ) -> np.array:
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


def explain_gradient_x_input(
    model: nn.Module,
    input_embeddings: torch.Tensor,
    y_batch: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    logits = model(None, attention_mask, inputs_embeds=input_embeddings).logits
    logits_for_class = get_logits_for_labels(logits, y_batch)
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    return torch.sum(grads * input_embeddings, dim=-1).detach()


def get_logits_for_labels(logits: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
    return logits[torch.arange(0, logits.shape[0], dtype=torch.int64), y_batch]
