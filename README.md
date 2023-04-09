<h1 align="center"><b>NoiseGrad and FusionGrad</b></h1>
<h3 align="center"><b>NoiseGrad: enhancing explanations by introducing stochasticity to model weights</b></h3>
<p align="center">
  <i>Pytorch implementation</i>
</p>

--------------

Pytorch implementation for **"NoiseGrad: enhancing explanations by introducing stochasticity to model weights"**. The paper introduces two novel methods `NoiseGrad` and `FusionGrad` which both improves attribution-based explanations by introducing stochasticity to the model parameters. See arXiv preprint: https://arxiv.org/abs/2106.10185.

![](https://raw.githubusercontent.com/understandable-machine-intelligence-lab/NoiseGrad/master/samples/resulting_explanation.png)

## Cite this paper

To cite this paper use following Bibtex annotation:

	@misc{bykov2021noisegrad,
	      title={NoiseGrad: enhancing explanations by introducing stochasticity to model weights},
	      author={Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne},
	      year={2021},
	      eprint={2106.10185},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}}

## Installation

To install requirements:

```setup
pip install noisegrad
```

All experiments were conducted with Python 3.6.9.

## Minimal Example, for more detailed ones, please refer to `examples/`
```python
from noisegrad import NoiseGrad, NoiseGradConfig, NoiseGradPlusPlus, NoiseGradPlusPlusConfig
from noisegrad.explainers import intgrad_explainer

# Initialize NoiseGrad: enhance any explanation function!
noisegrad = NoiseGrad(NoiseGradConfig(n=5))

# Initialize NoiseGrad++: enhance any explanation function!
noisegradp = NoiseGradPlusPlus(NoiseGradPlusPlusConfig(n=5, m=5))

# Get baseline explanation.
expl_base = intgrad_explainer(model, x, y)

# Get NoiseGrad explanation.
expl_ng = noisegrad.enhance_explanation(model, x, y, intgrad_explainer)

# Get NoiseGrad++ explanation.
expl_ngp = noisegradp.enhance_explanation(model, x, y, intgrad_explainer)
```


