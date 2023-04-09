from __future__ import annotations

from typing import Callable, Literal, NamedTuple, Protocol

import torch
import torch.nn as nn
from tqdm.auto import tqdm

NoiseType = Literal["multiplicative", "additive"]


class ExplanationFn(Protocol):

    def __call__(self, mode: nn.Module, x_batch: torch.Tensor, y_batch: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...


class NoiseGradConfig(NamedTuple):
    """
    mean:
        Mean of normal distribution, from which noise added to weights is sampled.
    std:
        Standard deviation normal distribution, from which noise added to weights is sampled.
    n:
        Number of types noise for weights is sampled.
    noise_type:
        Noise type, either multiplicative or additive, default=multiplicative.
    verbose:
            Indicates whether progress bar should be displayed, default=True.
    """

    mean: float = 1.0
    std: float = 0.2
    n: int = 10
    noise_type: NoiseType = "multiplicative"
    verbose: bool = True


class NoiseGradPlusPlusConfig(NamedTuple):
    """
    mean:
        Mean of normal distribution, from which noise added to weights is sampled.
    std:
        Standard deviation normal distribution, from which noise added to weights is sampled.
    sg_mean:
        Mean of normal distribution, from which noise added to inputs is sampled.
    sg_std:
        Standard deviation normal distribution, from which noise added to inputs is sampled.
    n:
        Number of types noise for weights is sampled.
    m:
        Number of types noise for inputs is sampled.
    noise_type:
        Noise type, either multiplicative or additive, default=multiplicative.
    verbose:
        Indicates whether progress bar should be displayed, default=True.
    """

    mean: float = 1.0
    std: float = 0.2
    sg_mean: float = 0.0
    sg_std: float = 0.4
    n: int = 10
    m: int = 10
    noise_type: NoiseType = "multiplicative"
    verbose: bool = True


class NoiseGrad:
    def __init__(self, config: NoiseGradConfig | None = None):
        """
        Parameters
        ----------
        config: Optional[NoiseGradConfig].
            Named tuple, as defined by NoiseGradConfig, if None, will use default values.

        """

        if config is None:
            config = NoiseGradConfig()

        self._std = config.std
        self._mean = config.mean
        self._n = config.n
        self._noise_type = config.noise_type
        self._verbose = config.verbose

        # Creates a normal (also called Gaussian) distribution.
        self._distribution = torch.distributions.normal.Normal(
            loc=self._mean, scale=self._std
        )

    def _sample(self, model: nn.Module):
        if self._std == 0.0:
            return

        # If std is not zero, loop over each layer and add Gaussian noise.
        with torch.no_grad():
            for layer in model.parameters():
                if self._noise_type == "additive":
                    layer.add_(self._distribution.sample(layer.size()).to(layer.device))
                else:
                    layer.mul_(self._distribution.sample(layer.size()).to(layer.device))

    def enhance_explanation(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        explanation_fn: ExplanationFn,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        model:
            Model which is subject to explanation.
        inputs:
            Batch of inputs, which is subject to explanations.
        targets:
            Batch of labels, which is subject to explanation.
        explanation_fn:
            Function used to generate baseline explanations. Takes model, batch of input, batch of labels as args.

        Returns
        -------

        ex:
            Enhanced explanations.
        """

        original_weights = model.state_dict().copy()
        explanation_shape = explanation_fn(model, inputs, targets).shape
        explanation = torch.zeros((self._n, *explanation_shape))

        it = tqdm(range(self._n), desc="NoiseGrad", disable=not self._verbose)
        for i in it:  # noqa
            self._sample(model)
            explanation[i] = explanation_fn(model, inputs, targets)

        model.load_state_dict(original_weights)
        return explanation.mean(axis=(0,))


class NoiseGradPlusPlus(NoiseGrad):
    def __init__(self, config: NoiseGradPlusPlusConfig | None = None):
        if config is not None:
            ng_config = NoiseGradConfig(
                n=config.n,
                mean=config.mean,
                std=config.std,
                verbose=False,
                noise_type=config.noise_type,
            )

        else:
            config = NoiseGradPlusPlusConfig()
            ng_config = NoiseGradConfig(verbose=False)

        super().__init__(ng_config)
        self._m = config.m
        self._sg_std = config.sg_std
        self._sg_mean = config.sg_mean
        self._verbose_pp = config.verbose

    def enhance_explanation(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        explanation_fn: ExplanationFn,
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        model:
            Model which is subject to explanation.
        inputs:
            Batch of inputs, which is subject to explanations.
        targets:
            Batch of labels, which is subject to explanation.
        explanation_fn:
            Function used to generate baseline explanations. Takes model, batch of input, batch of labels as args.
        Returns
        -------

        ex:
            Enhanced explanations.
        """

        explanation_shape = explanation_fn(model, inputs, targets).shape
        explanation = torch.zeros((self._m, self._n, *explanation_shape))
        original_weights = model.state_dict().copy()

        it = tqdm(
            range(self._n * self._m), desc="NoiseGrad++", disable=not self._verbose_pp
        )

        with it as pbar:
            for i in range(self._n):
                self._sample(model)
                for j in range(self._m):
                    noise = torch.randn_like(inputs) * self._sg_std + self._sg_mean
                    if self._noise_type == "additive":
                        inputs_noisy = inputs + noise
                    else:
                        inputs_noisy = inputs * noise
                    explanation[i][j] = explanation_fn(model, inputs_noisy, targets)
                    pbar.update()

        model.load_state_dict(original_weights)
        return explanation.mean(axis=(0, 1))
