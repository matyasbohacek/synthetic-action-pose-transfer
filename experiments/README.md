# Experiments

This module contains scripts for baseline and few-shot experiments described in our paper. These scripts train video classification models (in particular, of the TimeSformer, ViViT, and VideoMAE architectures) for the task of human action recognition.

## Getting Started

1. Follow the repository-level instructions to prepare a Python environment with the required packages.

2. Prepare your data in designated directories for real videos, synthetic videos with a white background, and synthetic videos with an image background. 

3. Specify command-line parameters (refer to in-code documentation and docstrings) and launch the training script with `python train.py`.

If you wish to reproduce results from our paper, refer to the shell scripts located in `./experiments/shell-scripts/`.