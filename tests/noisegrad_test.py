import functools

from noisegrad.explainers import explain_gradient_x_input, saliency_explainer
from noisegrad.noisegrad import NoiseGrad, NoiseGradPlusPlus


def test_noise_grad_image(
    resnet18_model, normalised_image, baseline_explanation, label, noise_grad_config
):
    ng = NoiseGrad(noise_grad_config)
    result = ng.enhance_explanation(
        resnet18_model, normalised_image, label, saliency_explainer
    )
    assert result.shape == baseline_explanation.shape


def test_noise_grad_pp_image(
    resnet18_model, normalised_image, baseline_explanation, label, noise_grad_pp_config
):
    ng_pp = NoiseGradPlusPlus(noise_grad_pp_config)

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
    noise_grad_config,
):
    ng = NoiseGrad(noise_grad_config)
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
    noise_grad_pp_config,
):
    ng_pp = NoiseGradPlusPlus(noise_grad_pp_config)
    explain_fn = functools.partial(
        explain_gradient_x_input, attention_mask=attention_mask
    )
    result = ng_pp.enhance_explanation(
        distilbert_model, input_embeddings, text_labels, explain_fn  # noqa
    )
    assert result.shape == baseline_explanation_text.shape
