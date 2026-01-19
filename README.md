# Component-level Knowledge Modeling for Task-Incremental Multimodal Learning

This repo is the official implementation of paper: Component-level Knowledge Modeling for Task-Incremental Multimodal Learning

## Abstract

Task-incremental learning for multimodal large models aims to enable models to continuously acquire new knowledge while preserving performance on previously learned tasks in a sequential learning scenario. However, new and old tasks often contain both task-specific knowledge and shared general knowledge. This structural overlap challenges existing LoRA-based expert methods in balancing knowledge preservation and knowledge expansion. To address this issue, we revisit multimodal task-incremental learning from a distribution modeling perspective. We represent each task as a global anchor pool composed of multiple Gaussian components. This representation distinguishes task-specific knowledge from shared knowledge at the component level. Based on this formulation, we design a cross-task component-level graph message passing mechanism. It enables shared knowledge components to interact and be refined while preserving task independence. During inference, we propose a taskagnostic expert fusion strategy without explicit task identifiers. The strategy adaptively fuses LoRA experts across tasks based on distributional consistency between the input sample and the global anchor pool. Experimental results demonstrate significant performance gains on both CoIN and UCIT benchmarks.
## Installation

```bash
conda create -n hide python=3.10 -y
conda activate hide
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
To measure the metrics of caption tasks, please install the following three packages:
```bash
pip install nltk==3.9.1
pip install pycocotools==2.0.8
pip install pycocoevalcap==1.2
```

## UCIT Benchmark

Please download the images from the constituting dataset：

|Image Source | Download Path|
| :-: | :-: |
|ArxivQA|[images](https://huggingface.co/datasets/MMInstruction/ArxivQA/tree/main)|
|ImageNet-R|[images](https://huggingface.co/datasets/HaiyangGuo/UCIT)|
|IconQA|[images](https://iconqa.github.io/)|
|CLEVR-Math|[images](https://huggingface.co/datasets/dali-does/clevr-math/tree/main)|
|VizWiz|[images](https://vizwiz.org/tasks-and-datasets/image-captioning/)|
|Flickr30k|[images](https://huggingface.co/datasets/HaiyangGuo/UCIT)|

After downloading all of them, organize the data as follows:
```
|-- datasets
    |-- ArxivQA
        |-- images/
    |-- CLEVR
        |-- images
            |-- train/
            |-- test/
            |-- val/
    |-- Flickr30k
        |-- train/
        |-- val/
    |-- IconQA
        |-- iconqa_data/
            |-- iconqa/
    |-- ImageNet-R
        |-- train/
        |-- test/
    |-- VizWiz
        |-- train/
        |-- test/
        |-- val/
```

Please download the instructions from UCIT's [HuggingFace](https://huggingface.co/datasets/HaiyangGuo/UCIT) page, then, organize the instructions as follows:
```
|-- instructions
    |-- ArxivQA
        |-- test_3000.json
        |-- train_4w.json
    |-- CLEVR
        |-- test_3000.json
        |-- train_4w.json
    |-- Flickr30k
        |-- test_3000.json
        |-- train_brief_4w.json
        |-- val_coco_type_3000.json
    |-- IconQA
        |-- test_3000.json
        |-- train.json
    |-- ImageNet-R
        |-- test_3000.json
        |-- train.json
    |-- VizWiz
        |-- test_3000.json
        |-- train.json
        |-- val_coco_type_3000.json
```

## Pre-trained Weights

Please download [LLaVA](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336), and use the **config.json** provided in this repository replace the original config.json in LLaVA.

## Training and Evaluation

Once the data and instructions organized and placed correctly, you can train the model by running `./scripts/CoIN/Train_UCIT/train_all.sh`. After the training is completed, you can evaluate the performance by running `./scripts/UCIT/Eval_UCIT/Eval_all.sh`. **Be careful to modify the paths in all `.sh` files to your own actual paths.**