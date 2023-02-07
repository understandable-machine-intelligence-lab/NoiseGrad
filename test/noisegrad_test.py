import functools

from noisegrad.explainers import saliency_explainer, explain_gradient_x_input
from noisegrad.noisegrad import NoiseGrad, NoiseGradPlusPlus


def test_noise_grad_image(
    resnet18_model, normalised_image, baseline_explanation, label
):
    ng = NoiseGrad(verbose=False, n=2)
    result = ng.enhance_explanation(
        resnet18_model, normalised_image, label, saliency_explainer
    )
    assert result.shape == baseline_explanation.shape


def test_noise_grad_pp_image(
    resnet18_model, normalised_image, baseline_explanation, label
):
    ng_pp = NoiseGradPlusPlus(verbose=False, n=2, m=2)

    result = ng_pp.enhance_explanation(
        resnet18_model, normalised_image, label, saliency_explainer
    )
    assert result.shape == baseline_explanation.shape


def test_noise_grad_text(
    distilbert_model,
    input_embeddings,
    attention_mask,
    baseline_explanation_text,
    text_labels,
):
    ng = NoiseGrad(verbose=False, n=2)
    explain_fn = functools.partial(
        explain_gradient_x_input, attention_mask=attention_mask
    )
    result = ng.enhance_explanation(
        distilbert_model, input_embeddings, text_labels, explain_fn  # noqa
    )
    assert result.shape == baseline_explanation_text.shape


def test_noise_grad_pp_text(
    distilbert_model,
    input_embeddings,
    attention_mask,
    baseline_explanation_text,
    text_labels,
):
    ng_pp = NoiseGradPlusPlus(verbose=False, n=2, m=2)
    explain_fn = functools.partial(
        explain_gradient_x_input, attention_mask=attention_mask
    )
    result = ng_pp.enhance_explanation(
        distilbert_model, input_embeddings, text_labels, explain_fn  # noqa
    )
    assert result.shape == baseline_explanation_text.shape
