# VGG16 Pruning with Reinforcement Learning

This project demonstrates the use of reinforcement learning (RL) to prune channels in a VGG16 model trained on the CIFAR-10 dataset. By pruning channels dynamically, the model achieves efficient compression while maintaining accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Training Results](#training-results)
- [References](#references)

## Introduction

Pruning channels in deep neural networks reduces computational costs and storage requirements. In this project, a policy gradient RL approach is employed to selectively prune channels in each layer of VGG16, optimized on the CIFAR-10 dataset.

Key highlights of the approach include:
- Using policy gradient-based REINFORCE algorithm.
- Batch-wise reward normalization.
- Fine-tuning of the model post-pruning.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision

Install dependencies using:
```bash
pip install torch torchvision
```

## Usage

### Clone the repository:

```bash
git clone https://github.com/yourusername/VGG16-RL-Pruning.git
cd VGG16-RL-Pruning
```

## Implementation Details

### Initial Training

The VGG16 model is first trained for a few epochs on the CIFAR-10 dataset without pruning to establish baseline accuracy.

### RL-based Pruning

Using the REINFORCE algorithm, we iteratively prune channels based on learned policies for each convolutional layer. The pruning strategy relies on:
- A sigmoid transformation of learnable parameters for each channel, producing a Bernoulli mask.
- A reward structure combining compression with accuracy incentives to balance sparsity and performance.

### Fine-Tuning

After pruning, the model is fine-tuned to recover accuracy with a reduced set of channels.

### Key Functions

- **LayerAgent**: Encapsulates the policy gradient logic for channel pruning.
- **PruningEnvironment**: Manages the generation of masks and updates the model during RL-based pruning.

## Training Results

Below is a sample of expected output after successful pruning:

- **Initial Accuracy**: ~91%
- **Final Accuracy after Pruning and Fine-Tuning**: ~90%
- **Pruned Channels**: Count of channels set to zero across layers, demonstrating compression.

