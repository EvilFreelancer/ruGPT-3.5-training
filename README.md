# Training ruGPT-3.5 with LoRA and converting to GGML

Collection of scripts and Jupyter Notebook with comprehensive scripts for
training custom ruGPT-3.5 on datasets from Saiga-2 (rulm).

> This model was trained using settings identical to [GigaSaiga](https://huggingface.co/IlyaGusev/gigasaiga_lora),
> but incorporates two additional datasets.

## Training

By following this instruction you will be able to train your own ruGPT-3.5 model with help of Peft/LoRA
using [Saiga-2](https://github.com/IlyaGusev/rulm) datasets, but of course you may add your own datasets.

System requirements:

* ~100 GB of system RAM
* ~200 GB on HDD/SSD
* Nvidia GPU with 20+ GB VRAM (eg. RTX 3090 or 4090)
* CUDA 12.2

### Install libraries manually

Requirements:

* Python 3.10
* Python VirtualEnv

Clone the repo with all submodules:

```shell
git clone --recurse-submodules https://github.com/EvilFreelancer/ruGPT-3.5-training.git
```

Instantiate a virtual environment:

```shell
python -m venv venv
```

Switch to a virtual environment:

```shell
source venv/bin/activate
```

Download Python packages:

```shell
pip install -r requirements.txt
```

### Install libraries in Docker

Requirements:

* Docker
* Docker Compose
* Nvidia Docker Runtime

Solution based
on [nvidia/cuda:12.2.0-devel-ubuntu22.04](https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=12.2.0-devel-ubuntu22.04)
image.

Clone the repo with all submodules:

```shell
git clone --recurse-submodules https://github.com/EvilFreelancer/ruGPT-3.5-training.git
```

Copy compose config from dist (and change settings if you need):

```shell
cp docker-compose.dist.yml docker-compose.yml
```

Build an image:

```shell
docker-compose build
```

Start container:

```shell
docker-compose build
```

Attach to container's shell:

```shell
docker-compose exec app bash
```

### Datasets

The datasets utilized for training this model are consistent with those used
for [Saiga-2 (rulm)](https://github.com/IlyaGusev/rulm).

Here's the comprehensive list:

- [ru_turbo_alpaca](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca)
- [ru_turbo_alpaca_evol_instruct](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca_evol_instruct)
- [ru_turbo_saiga](https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga)
- [ru_sharegpt_cleaned](https://huggingface.co/datasets/IlyaGusev/ru_sharegpt_cleaned)
- [oasst1_ru_main_branch](https://huggingface.co/datasets/IlyaGusev/oasst1_ru_main_branch)
- [gpt_roleplay_realm](https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm)
- [ru_instruct_gpt4](https://huggingface.co/datasets/lksy/ru_instruct_gpt4)

To download and merge all datasets from this list you need to execute:

```shell
python3 1_dataset.py
```