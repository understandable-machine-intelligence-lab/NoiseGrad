from typing import Callable
import torch


class NoiseGrad:
    def __init__(
        self,
        model,
        weights,
        mean: float = 1.0,
        std: float = 0.2,
        n: int = 25,
        noise_type: str = "multiplicative",
    ):
        """
        Initialize the explanation-enhancing method: NoiseGrad.
        Paper:

        Args:
            model (torch model): a trained model
            weights (dict):
            mean (float): mean of the distribution (often referred to as mu)
            std (float): standard deviation of the distribution (often referred to as sigma)
            n (int): number of Monte Carlo rounds to sample models
            noise_type (str): the type of noise to add to the model parameters, either additive or multiplicative
        """

        self.std = std
        self.mean = mean
        self.model = model
        self.n = n
        self.weights = weights
        self.noise_type = noise_type

        # Creates a normal (also called Gaussian) distribution.
        self.distribution = torch.distributions.normal.Normal(
            loc=self.mean, scale=self.std
        )

        print("NoiseGrad initialized.")

    def sample(self):
        self.model.load_state_dict(self.weights)
        # If std is not zero, loop over each layer and add Gaussian noise.
        if not self.std == 0.0:
            with torch.no_grad():
                for layer in self.model.parameters():
                    if self.noise_type == "additive":
                        layer.add_(
                            self.distribution.sample(layer.size()).to(layer.device)
                        )
                    elif self.noise_type == "multiplicative":
                        layer.mul_(
                            self.distribution.sample(layer.size()).to(layer.device)
                        )
                    else:
                        print(
                            "Set NoiseGrad attribute 'noise_type' to either 'additive' or 'multiplicative' (str)."
                        )

    def enhance_explanation(self, inputs, targets, explanation_fn: Callable, **kwargs):
        """Sample explanation."""
        explanation = torch.zeros(
            (self.n, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        )

        for i in range(self.n):
            self.sample()
            explanation[i] = explanation_fn(
                self.model.to(kwargs.get("device", None)), inputs, targets
            )

        return explanation.mean(axis=(0))


class NoiseGradPlusPlus(NoiseGrad):
    def __init__(
        self,
        model,
        weights,
        mean: float = 1.0,
        std: float = 0.2,
        sg_mean: float = 0.0,
        sg_std: float = 0.4,
        n: int = 10,
        m: int = 10,
        noise_type: str = "multiplicative",
    ):
        """
        Initialize the explanation-enhancing method: NoiseGrad++.
        Paper:

        Args:
            model (torch model): a trained model
            weights (dict):
            mean (float): mean of the distribution (often referred to as mu)
            std (float): standard deviation of the distribution (often referred to as sigma)
            n (int): number of Monte Carlo rounds to sample models
            noise_type (str): the type of noise to add to the model parameters, either additive or multiplicative

        Args:
            model:
            weights:
            mean:
            std:
            sg_mean:
            sg_std:
            n:
            m:
            noise_type:
        """

        self.std = std
        self.mean = mean
        self.model = model
        self.n = n
        self.m = m
        self.sg_std = sg_std
        self.sg_mean = sg_mean
        self.weights = weights
        self.noise_type = noise_type

        # Creates a normal (also called Gaussian) distribution.
        self.distribution = torch.distributions.normal.Normal(
            loc=self.mean, scale=self.std
        )

        super(NoiseGrad, self).__init__()
        print("NoiseGrad++ initialized.")

    def sample(self):
        self.model.load_state_dict(self.weights)
        # If std is not zero, loop over each layer and add Gaussian noise.
        if not self.std == 0.0:
            with torch.no_grad():
                for layer in self.model.parameters():
                    if self.noise_type == "additive":
                        layer.add_(
                            self.distribution.sample(layer.size()).to(layer.device)
                        )
                    elif self.noise_type == "multiplicative":
                        layer.mul_(
                            self.distribution.sample(layer.size()).to(layer.device)
                        )
                    else:
                        print(
                            "Set NoiseGrad attribute 'noise_type' to either 'additive' or 'multiplicative' (str)."
                        )

    def enhance_explanation(self, inputs, targets, explanation_fn: Callable, **kwargs):
        """Sample explanation."""
        explanation = torch.zeros(
            (self.n, self.m, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        )

        for i in range(self.n):
            self.sample()
            for j in range(self.m):
                inputs_noisy = (
                    inputs + torch.randn_like(inputs) * self.sg_std + self.sg_mean
                )
                explanation[i][j] = explanation_fn(
                    self.model.to(kwargs.get("device", None)), inputs_noisy, targets
                )

        return explanation.mean(axis=(0, 1))
