<div align="center">

# αβ-GAN: Robust generative adversarial networks

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Abstract
>Generative adversarial networks (GAN) training is subject to problems including mode collapse, gradient vanishing, and instability. Although many different losses have been proposed to alleviate these shortcomings, they heavily rely on a fixed-value function with limited expressive power in terms of robustness, whereby failing to perform consistently over multiple data sets. To solve this problem, we propose a parametric and robust αβ-loss function that can improve the performances of GAN on different data sets. Specifically, unlike standard GAN loss function it exploits the αβ-divergence (AB-divergence) to weigh the likelihood ratio associated with each data point. This weighing mechanism makes the model robust to noises and yields better models in terms of FID score. To reduce the cost of searching for the optimal and, we further propose an adaptive version to systematically update these parameters according to statistics of the discriminator’s output. Moreover, αβ-loss can be reduced to Least-Square GAN (LS-GAN) and standard GAN (SGAN) loss function as special cases. We conduct extensive experiments on both synthetic and real-world data sets. Experimental results over the synthetic data sets (2D Gaussian ring and grid) demonstrate that our approach can significantly alleviate the issue of mode collapse. Additionally, by constraining the gradient of the discriminator that is fed back to the generator via finely adjusting the hyper-parameters α and β, our approach can improve the quality of synthetic images, as can be seen from the decrease of FID from 40 to 23.71 on the data set CIFAR10 using the SN-DCGAN architecture.

## Getting started
After cloning this repo, install dependencies
```yaml
# [OPTIONAL] create conda environment
conda create --name abGAN-py38 python=3.8
conda activate abGAN-py38

# install requirements
pip install -r requirements.txt
```

Train model with experiment configuration
```yaml
# default
python run.py experiment=train_cifar10_gan.yaml

# train on CPU
python run.py experiment=train_cifar10_gan.yaml trainer.gpus=0

# train on GPU
python run.py experiment=train_cifar10_gan.yaml trainer.gpus=1
```

You can override any parameter from command line like this.
```yaml
python run.py experiment=train_cifar10_gan.yaml trainer.max_epochs=200 datamodule.batch_size=32
```
## 📜  License

This project is licensed under the Apache 2.0 License – see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute αβ-GAN in either commercial or academic projects under the terms of this license.

## 📚 Citation

If you use **αβ-GAN** in your research or applications, please consider citing our paper:

```bibtex
@article{GNANHA2022177,
title = {αβ-GAN: Robust generative adversarial networks},
journal = {Information Sciences},
volume = {593},
pages = {177-200},
year = {2022},
issn = {0020-0255},
doi = {https://doi.org/10.1016/j.ins.2022.01.073},
url = {https://www.sciencedirect.com/science/article/pii/S0020025522001128},
author = {Aurele Tohokantche Gnanha and Wenming Cao and Xudong Mao and Si Wu and Hau-San Wong and Qing Li},
keywords = {Image synthesis, Generative adversarial networks, Deep learning},
}
```
