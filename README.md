
# Efficient Resource-Constrained Training of Transformers via Subspace Optimization

> Le-Trung Nguyen, Enzo Tartaglione, Van-Tam Nguyen

Implementation for the paper `Efficient Resource-Constrained Training of Transformers via Subspace Optimization`.

<details><summary>Abstract</summary>

In today’s world, where AI plays a major role in everyday life, energy consumption and data privacy have become critical concerns. On-device learning offers a promising solution by enabling models to train directly on edge devices, thereby reducing energy usage and minimizing the risk of data leakage. However, the increasing size of modern neural networks poses a serious challenge for on-device training. Although prior work has mainly focused on compact convolutional architectures, we explore a different direction by applying subspace-based training to transformer models. Based on the idea that a model’s essential information resides in a fixed subspace, we introduce Weight-Activation Subspace Iteration (WASI), a method designed to overcome the memory bottleneck of backpropagation and improve inference efficiency in transformer-based models by constraining training to this subspace. Our results show that, with accuracy comparable to vanilla training, WASI reduces memory usage by up to $62\times$ and computational cost (FLOPs) by up to $2\times$. Moreover, when tested on a Raspberry Pi 5, WASI delivers approximately $1.4\times$ faster training and inference than vanilla training.

</details>


## Environment Setup
> Experiments were conducted on Ubuntu 22.04.4 LTS.

1. Install the Miniconda environment by following the instructions [here](https://docs.anaconda.com/miniconda/)

2. Create and activate conda virtual environment

```
conda create -n wasi python=3.8
conda activate wasi
```
3. Install requirements

```
pip install -r requirements.txt
```
4. Install CUDA 11.7

```
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
```

5. Install PyTorch 1.13.1

```
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### Main experiments

0. Make sure to be inside [main](main/) folder

```
cd main
```

1. Install dependencies

```
pip install "jsonargparse[signatures]" pytorch_lightning==1.6.5 torchmetrics==0.9.2 pretrainedmodels
```

2. Prepare checkpoints for SVD-LLM

```
gdown https://drive.google.com/uc?id=1Z25gmtIYYsXqu5q-gUu1pJdc1FDowbZX
unzip SVD_LLM_checkpoints.zip
rm SVD_LLM_checkpoints.zip
```

3. Run experiments

In the [scripts](main/scripts) directory, you can find pre-configured bash files to run the corresponding experiments. For example, to fine-tune ViT and SwinT with WASI:

```bash
bash  scripts/WASI/WASI.sh
```

The output file will be stored in the `./runs` folder.


  

### On Device Latency

0. Make sure to be inside [on_device_latency](on_device_latency/) folder

```
cd on_device_latency
```

1. To run the experiment, use:

```bash
bash  scripts/test_cpu.sh
```

The output will be stored in `processed_time_cpu/`.

## References
If you use this code, please cite this work as:
```
@inproceedings{
nguyen2026efficient,
title={Efficient Resource-Constrained Training of Transformers via Subspace Optimization},
author={Le-Trung Nguyen and Enzo Tartaglione and Van-Tam Nguyen},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026}
}
```