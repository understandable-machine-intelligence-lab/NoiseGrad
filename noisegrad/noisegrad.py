from __future__ import annotations

from typing import Callable
import torch
from tqdm.auto import tqdm


ExplanationFn = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


class NoiseGrad:
    def __init__(
        self,
        mean: float = 1.0,
        std: float = 0.2,
        n: int = 10,
        noise_type: str = "multiplicative",
        verbose: bool = True,
    ):
        """

        Parameters
        ----------
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

        self._std = std
        self._mean = mean
        self._n = n
        self._noise_type = noise_type
        self._verbose = verbose

        # Creates a normal (also called Gaussian) distribution.
        self._distribution = torch.distributions.normal.Normal(
            loc=self._mean, scale=self._std
        )

    def _sample(self, model: torch.nn.Module):
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
        model: torch.nn.Module,
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

        original_weights = model.state_dict()
        explanation_shape = explanation_fn(model, inputs, targets).shape
        explanation = torch.zeros((self._n, *explanation_shape))

        it = tqdm(range(self._n), desc="NoiseGrad", disable=not self._verbose)
        for i in it:  # noqa
            self._sample(model)
            explanation[i] = explanation_fn(model, inputs, targets)

        model.load_state_dict(original_weights)
        return explanation.mean(axis=(0,))


class NoiseGradPlusPlus(NoiseGrad):
    def __init__(
        self,
        mean: float = 1.0,
        std: float = 0.2,
        sg_mean: float = 0.0,
        sg_std: float = 0.4,
        n: int = 10,
        m: int = 10,
        noise_type: str = "multiplicative",
        verbose: bool = True,
    ):
        """

        Parameters
        ----------
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

        super().__init__(mean, std, n, noise_type, False)
        self._m = m
        self._sg_std = sg_std
        self._sg_mean = sg_mean
        self._verbose_pp = verbose

    def enhance_explanation(
        self,
        model: torch.nn.Module,
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
        original_weights = model.state_dict()

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
