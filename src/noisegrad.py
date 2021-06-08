import torch

class NoiseGrad:
    def __init__(self, mean: float = 1.0,
                 std: float = 0.4,
                 type: str = "multiplicative"):
        """
        NoiseGrad wrapper.

        Args:
            mean (float): mean of the distribution (often referred to as mu)
            std (float): standard deviation of the distribution (often referred to as sigma)
            type (str): the type of noise to add to the model parameters, either additive or muliplticative
        """
        self.mean = mean
        self.std = std
        self.type = type

        # Creates a normal (also called Gaussian) distribution.
        self.distribution = torch.distributions.normal.Normal(loc=self.mean, scale=self.std)

    def __call__(self, model, weights):
        model.load_state_dict(weights)

        # If std is not zero, loop over each layer and add Gaussian noise.
        if not self.std == 0.0:
            with torch.no_grad():
                for layer in model.parameters():
                    if self.type == "additive":
                        layer.add_(self.distribution.sample(layer.size()).to(layer.device))
                    elif self.type == "multiplicative":
                        layer.mul_(self.distribution.sample(layer.size()).to(layer.device))
                    else:
                        print("Set NoiseGrad attribute 'type' to either 'additive' or 'multiplicative' (str).")


class NoiseGradplusplus:
    def __init__(self, mean: float = 1.0,
                 std: float = 0.4,
                 type: str = "multiplicative"):
        """
        NoiseGrad wrapper.

        Args:
            mean (float): mean of the distribution (often referred to as mu)
            std (float): standard deviation of the distribution (often referred to as sigma)
            type (str): the type of noise to add to the model parameters, either additive or muliplticative
        """
        self.mean = mean
        self.std = std
        self.type = type

        # Creates a normal (also called Gaussian) distribution.
        self.distribution = torch.distributions.normal.Normal(loc=self.mean, scale=self.std)

    def __call__(self, model, weights):
        model.load_state_dict(weights)

        # If std is not zero, loop over each layer and add Gaussian noise.
        if not self.std == 0.0:
            with torch.no_grad():
                for layer in model.parameters():
                    if self.type == "additive":
                        layer.add_(self.distribution.sample(layer.size()).to(layer.device))
                    elif self.type == "multiplicative":
                        layer.mul_(self.distribution.sample(layer.size()).to(layer.device))
                    else:
                        print("Set NoiseGrad attribute 'type' to either 'additive' or 'multiplicative' (str).")
