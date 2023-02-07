"""Some examples of explanation methods that can be used in NosieGrad and NoiseGrad++ implementations."""
from __future__ import annotations

import torch
from captum.attr import Saliency, IntegratedGradients
from typing import Optional
import numpy as np


def saliency_explainer(
    model: torch.nn.Module, inputs, targets, normalize: bool = False, **kwargs
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
            inputs.shape[0], kwargs.get("img_size", 224), kwargs.get("img_size", 224)
        )
        .cpu()
        .data
    )

    if normalize:
        return normalize_heatmap(explanation)

    return explanation


def intgrad_explainer(
    model: torch.nn.Module,
    inputs,
    targets,
    abs: bool = True,
    normalize: bool = False,
    **kwargs
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
            inputs.shape[0], kwargs.get("img_size", 224), kwargs.get("img_size", 224)
        )
        .cpu()
        .data
    )


def explain_gradient_x_input(
    model: torch.nn.Module,
    input_embeddings: torch.Tensor,
    y_batch: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:

    logits = model(None, attention_mask, inputs_embeds=input_embeddings).logits
    indexes = torch.reshape(y_batch, (len(y_batch), 1)).to(torch.int64)
    logits_for_class = torch.gather(logits, dim=-1, index=indexes)
    grads = torch.autograd.grad(torch.unbind(logits_for_class), input_embeddings)[0]
    return torch.sum(grads * input_embeddings, dim=-1).detach()
