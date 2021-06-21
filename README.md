<h1 align="center"><b>NoiseGrad and NoiseGrad++</b></h1>
<h3 align="center"><b>NoiseGrad: enhancing explanations by introducing stochasticity to model weights</b></h3>
<p align="center">
  <i>Pytorch implementation</i>
</p> 
 
--------------

Pytorch implementation for **"NoiseGrad: enhancing explanations by introducing stochasticity to model weights"**. The paper introduces two novel methods `NoiseGrad` and `NoiseGrad++` which both improves attribution-based explanations by introducing stochasticity to the model parameters. See arXiv preprint: https://arxiv.org/abs/2106.10185.

<p align="center">
  <img src="samples/resulting_explanation.png" alt="Visualization of baseline, NoiseGrad and NoiseGrad++ explanations using (Integrated Gradient) as XAI method." width="512"/>  
</p>

## Cite this paper

To cite this paper use following Bibtex annotation:

	@misc{bykov2021noisegrad,
	      title={NoiseGrad: enhancing explanations by introducing stochasticity to model weights}, 
	      author={Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne},
	      year={2021},
	      eprint={2106.10185},
	      archivePrefix={arXiv},
	      primaryClass={cs.LG}}

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

All experiments were conducted with Python 3.6.9.

## Code structure

The source code can be found in the `src/` folder and an example notebook in `examples/` folder.

