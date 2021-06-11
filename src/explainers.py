"""Some examples of explanation methods that can be used in NosieGrad and NoiseGrad++ implementations."""
from captum.attr import *
from .utils import *


def saliency_explainer(model, inputs, targets, normalize=False, **kwargs) -> np.array:
    """Implementation of InteGrated Gradients by Sundararajan et al., 2017 using Captum."""

    assert (
        len(np.shape(inputs)) == 4
    ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

    explanation = (
        Saliency(model)
        .attribute(inputs, targets, abs=True)
        .sum(axis=1)
        .reshape(kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        .cpu()
        .data
    )

    if normalize:
        return normalize_heatmap(explanation)

    return explanation


def intgrad_explainer(
    model, inputs, targets, abs=True, normalize=False, **kwargs
) -> np.array:
    """Implementation of InteGrated Gradients by Sundararajan et al., 2017 using Captum."""

    assert (
        len(np.shape(inputs)) == 4
    ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

    explanation = IntegratedGradients(model).attribute(
        inputs=inputs, target=targets, baselines=torch.zeros_like(inputs)
    )
    if abs:
        explanation = explanation.abs()
    if normalize:
        explanation = normalize_heatmap(explanation)

    return (
        explanation.sum(axis=1)
        .reshape(kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        .cpu()
        .data
    )
