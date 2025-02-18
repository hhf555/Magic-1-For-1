<div align="center">

# Magic 1-For-1: Generating One Minute Video Clips within One Minute

</div>
<div align="center">
    <a href="https://magic-141.github.io/Magic-141/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
</div>

## 📖 Overview

**Magic 1-For-1** is an efficient video generation model designed to optimize memory usage and reduce inference latency. It decomposes the text-to-video generation task into two sub-tasks: **text-to-image generation** and **image-to-video generation**, enabling more efficient training and distillation.

### Updates
- **$\texttt{[2025-02.15]}$:** We are working on some clearances to release additional information for **Magic 1-For-1**. We appreciate your understanding and will share updates soon.
- **$\texttt{[2025-02.07]}$:** We have released code for **Magic 1-For-1**.
<!-- - 🚀 More to Come!
We are continuously working on improving and expanding the capabilities of **Magic 1-For-1**. Contributions and collaborations are welcome! Join us in advancing the field of **interactive foundation video generation**. -->

## 📹 Demo


https://github.com/user-attachments/assets/94069a93-b2bb-4900-84f7-ca7603c04ecc


## 🛠️ Preparations

### Environment Setup
First, make sure **git-lfs** is installed (https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

It's recommended to use conda to manage the project's dependencies. First, it is needed to create a conda environment named video-generation and specify the Python version.
```bash
conda create -n video_infer python=3.9  # Or your preferred Python version
conda activate video_infer
```
The project's dependencies are listed in the requirements.txt file. You can use pip to install all dependencies at once.
```bash
pip install -r requirements.txt
```

### 📥 Downloading Model Weights (need to finish clearances)

<!-- > ⚠️ **WARNING**: Flash Attention 3 is required for inference to avoid CUDA errors, by including `export USE_FLASH_ATTENTION3=1` -->

## 🚀 Inference 
### Video Generation (Single GPU)
Magic 1-For-1 supports two modes of video generation:
1. Text to Video

The script uses argparse to handle command-line arguments.  Here's a breakdown:

`-c`, `--config`: Path to the configuration file (required for t2v).
`-q`, `--quantization`: Enable quantization (optional, defaults to False).
`-l`, `--quantization_level`: Quantization level (e.g., "int8", optional, defaults to "int8").
`-m`, `--low_memory`: Enable low memory mode (optional, defaults to False).
`--cli`: Enable command-line interactive mode (optional, defaults to False).

For Text to Video generation, run the following command:

```bash
python test_t2v.py --config configs/test/text_to_video/4_step_t2v.yaml --quantization False 
```

Alternatively, use the provided script:

```bash
bash scripts/run_t2v.sh
```

### 💻 Quantization
This project enables quantization techniques to optimize model performance, reduce memory footprint, and accelerate inference. We support various quantization levels, including INT8 and INT4, and provide options for using either Optimum-Quanto or TorchAO for quantization.

1. Install dependencies:

```bash
pip install optimum-quanto torchao
```

2. Usage

To enable quantization, set the `--quantization` flag to `True` when running your script.

```bash
python test_t2v.py --config configs/test/4_step_t2v.yaml --quantization True
```

Specify the quantization level using the `--quantization_level` flag. Available options are `int8` and `int4`. 

3.1 INT8 Quantization with Optimum-Quanto

```bash
python test_t2v.py --config configs/test/4_step_t2v.yaml --quantization True --quantization_level int8
```

3. Additional Notes

Put the quantization file after `outputs/quant` to enable loading the pre-quanted weights. You can refer to the documentation of Optimum-Quanto and TorchAO for more advanced quantization techniques and options.

### 🖥️ Multi-GPU Inference

To run inference on multiple GPUs, specify the number of GPUs and their IDs. Adjust the `ring_degree` and `ulysses_degree` values in the configuration file to match the number of GPUs used.

text to video
```bash
    bash scripts/run_flashatt3.sh test_t2v.py configs/test/t2v.yaml 1 0
```


## 📃 Citation

Please cite the following paper when using this model:
```bash
@article{yi2025magic,
  title={Magic 1-For-1: Generating One Minute Video Clips within One Minute},
  author={Hongwei Yi, Shitong Shao, Tian Ye, Jiantong Zhao, Qingyu Yin, Michael Lingelbach, Li Yuan, Yonghong Tian, Enze Xie, Daquan Zhou},
  journal={to be updated},
  year={2025}
}
```

