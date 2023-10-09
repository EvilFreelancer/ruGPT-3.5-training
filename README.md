# Training ruGPT-3.5 13B with LoRA

This repository provides a curated collection of scripts and a Jupyter Notebook designed for training a custom
`ruGPT-3.5-13B` model in the `load_in_8bit` mode utilizing datasets and some scripts
from [Saiga-2 (rulm)](https://github.com/IlyaGusev/rulm).

The training process outlined here leverages Peft/LoRA technology. The resources provided are designed to facilitate a
smooth training experience, seamless merging of LoRA weights with the original model, and a straightforward conversion
of the model into the GGML format.

> Note: While the training settings for this model mirror those used
> in [GigaSaiga](https://huggingface.co/IlyaGusev/gigasaiga_lora), but my model is enriched with
> additional dataset.

The primary objective of this repository is to reproduce the success achieved by GigaSaiga and to provide a detailed,
step-by-step documentation of the training procedure. This initiative aims to empower and support the Russian-speaking
ML community by making the process of training the ruGPT-3.5-13B model more accessible and understandable.

For your convenience, pretrained models are readily available at the following locations:

- https://huggingface.co/evilfreelancer/ruGPT-3.5-13B-lora
- https://huggingface.co/evilfreelancer/ruGPT-3.5-13B-ggml

By following the instructions and using the scripts provided in this repository, users can efficiently train their
versions of the ruGPT-3.5-13B model with the flexibility to incorporate additional datasets as necessary.

## Acknowledgments

First of all I would like to extend our sincere gratitude to the following authors and contributors:

- The [Sber AI](https://ai.sber.ru/) Team, the brains behind the original `ruGPT-3.5-13B` model. Their groundbreaking
  work
  and continuous efforts in advancing AI and machine learning technologies have laid a solid foundation for this project
  and many others in the AI community.

- [IlyaGusev](https://github.com/IlyaGusev) and the [rulm](https://github.com/IlyaGusev/rulm) project team for their
  invaluable resources and datasets from Saiga-2/GigaSaiga, which have been fundamental in the training process of this
  custom ruGPT-3.5-13B model.

- [graysonwhite](https://github.com/graysonwhite) and the [gglm](https://github.com/graysonwhite/gglm) project team. I'm
  particularly thankful for their comprehensive documentation on the ggml project, which has been indispensable in
  guiding me through the correct procedures for model transformation.

- [iashchak](https://huggingface.co/iashchak) for
  his [ruGPT-3.5-13B-ggml](https://huggingface.co/iashchak/ruGPT-3.5-13B-ggml) repository on HuggingFace. His
  contributions and shared expertise with [llm-rs-python](https://github.com/LLukas22/llm-rs-python) have been crucial
  in the successful creation of this project.

This project has been significantly enriched and made possible through the cumulative efforts and shared knowledge of
these incredible individuals and teams. I deeply appreciate their contributions and are immensely thankful for their
openness to sharing resources with the broader community.

For anyone looking to understand, extend, or build upon my work, I strongly recommend referring to and acknowledging
these original authors and contributors, as their work represents the cornerstone of this project and many others
in the field.

## Global requirements

Before embarking on the training process, ensure your system meets the following requirements:

* ~100 GB of system RAM
* ~200 GB on HDD/SSD
* Nvidia GPU with at least 20 GB VRAM (eg. RTX 3090 or 4090)
* CUDA 12.2

## Install libraries manually

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

## Install libraries in Docker

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

## Training with LoRA

The entire process is broken down into four main steps, each corresponding to a
script in the project’s root directory. Below is a step-by-step guide.

### Step 1: Dataset Download and Merging

```shell
python3 1_dataset.py
```

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

The resultant datasets `train_full.jsonl` and `val_full.jsonl` are generated in chat format.

### Step 2: Model Training LoRA

```shell
python3 2_train.py
```

The sequence of operations performed by this script is as follows:

1. **Download Original Model**: The script initiates by downloading the original `ruGPT-3.5-13B` model from HuggingFace.
   The downloaded files are stored in the `ruGPT-3.5-13B` folder.

2. **Configuration Modification**: After the download is complete, the script copies and modifies the configuration
   files located in `ruGPT-3.5-13B` folder. The altered configurations, which are necessary to enable training, are then
   placed in the `output` folder.

3. **Training Initialization**: The script subsequently instantiates the `src.train` Python module from the `rulm`
   project. This operation occurs within the `rulm/self_instruct` subdirectory.

4. **Output Files**: Upon the completion of the above steps, `adapter_model.bin` and `adapter_config.json` are generated
   and saved in the `output` folder.

Each of the generated files plays a crucial role in the subsequent steps of the model training and application process.

### Step 3: Model Merging

```shell
python3 3_merge.py
```

This script performs the following tasks:

- **Weights Merging**: It uses a modified version of the [convert_to_native.py] script. The script seamlessly merges the
  LoRA adapter weights with the weights of the base `ruGPT-3.5-13B` model. This merging process is crucial for enhancing
  the model’s performance with the learned adaptations from the LoRA training.

- **Saving Merged Model**: After the merging process is complete, the script saves the resultant model with the filename
  `pytorch_model.bin` in the `output` directory of project.

Ensure you have sufficient storage space available in the `output` directory as the merged model file can be quite
large.

### Step 4: GGML Conversion

```shell
python3 4_ggml.py
```

This step involves two main tasks:

1. **Conversion to GGML-Compatible Format**: The script starts by converting the `pytorch_model.bin` file into a format
   that is compatible with GGML. This converted format serves as an intermediate step that prepares the model for
   subsequent quantization processes.

2. **Quantization**: Following the initial conversion, the script performs quantization on the model. The quantization
   process generates various quantized versions of the model, specifically: q4_0, q4_1, q5_0, q5_1, and q8_0. Each
   quantized version is optimized for different levels of precision and performance requirements.

3. **Library Utilization**: This entire process utilizes the [llm-rs-python](https://github.com/LLukas22/llm-rs-python)
   library. Ensure that this library is installed and accessible, as it plays a pivotal role in the GGML conversion and
   quantization processes.

4. **Saving GGML Models**: Upon completion of the conversion and quantization steps, the script saves the resultant GGML
   models in the `output_ggml` directory within your project’s root.

Ensure you have adequate storage space available in the `output_ggml` directory, as the GGML models, especially the
quantized versions, may occupy significant space.

## Testing Scripts

The root directory contains four additional scripts for testing each intermediate step:

- **test_gigasaiga.py**: Demonstrates the functionality of the original GigaSaiga as implemented by the authors of the
  rulm project.
- **test_lora.py**: Tests the on-the-fly merging of the LoRA adapter with adapter_model.bin from the output directory.
- **test_merged.py**: Shows the functionality of the original ruGPT-3.5 model after LoRA weights merging.
- **test_ggml.py**: Tests the GGML versions of the model to ensure proper functioning.

## Support and Contribution

Feel free to open issues or pull requests if you have suggestions or encounter issues.
Contributions to improve or expand this project are always welcome!

## Links

- https://huggingface.co/ai-forever/ruGPT-3.5-13B
- https://github.com/IlyaGusev/rulm
- https://github.com/ggerganov/ggml
- https://github.com/LLukas22/llm-rs-python
- https://huggingface.co/iashchak/ruGPT-3.5-13B-ggml
- https://huggingface.co/iashchak/ruGPT-3.5-13B-ggml/discussions/3
