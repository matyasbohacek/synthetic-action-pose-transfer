
<img width="1035" alt="arxiv-teaser" src="https://github.com/user-attachments/assets/2a8d0f2e-7993-45d5-94df-5c6deb226b68" />

# Pose-Transfer Synthetic Data Generation for Video Model Training

#### [Vaclav Knapp](https://www.linkedin.com/in/václav-knapp-7696b624a/) and [Matyas Bohacek](https://www.matyasbohacek.com)

In video understanding tasks, particularly those involving human actions, synthetic data generation often suffers from uncanny features, diminishing its impact on training. Tasks such as sign language translation, gesture recognition, and human motion understanding in autonomous driving have thus been unable to exploit the full potential of synthetic data. This paper proposes a method for generating high-fidelity synthetic training data of human action videos based on controllable 3D Gaussian avatar models. We evaluate this method on a subset of the Toyota Smarthome dataset and show that it improves performance in action recognition tasks. Moreover, we demonstrate that the method can effectively scale few-shot datasets and improve robustness by increasing their size, making up for groups underrepresented in the real training data, and adding diverse backgrounds. We open-source the method, along with a dataset of synthetic videos based on the Smarthome dataset, which we call _RANDOM People_.

> [See paper]() — See poster — [Contact us](mailto:maty-at-stanford-dot-edu)
> 
> _Pre-print released on arXiv_

## Getting Started

TBD

## Data

If you want to use our _RANDOM People_ dataset, refer to [its Hugging Face page](https://huggingface.co/datasets/matybohacek/RANDOM-People). If you wish to reproduce our results on the Toyota Smarthome dataset, please refer to [the official distribution instructions](https://project.inria.fr/toyotasmarthome/), as presented by its authors.

## Features

[**Synthetic Data Generation.**](synthetic_data_generation/) We open-source our method for synthetic video data generation built around pose transfer, used to create the _RANDOM People_ dataset. This method can be used to scale an arbitrary dataset with videos of human action.

[**Experiments.**](experiments/) This repository also includes scripts for baseline and few-shot experiments described in our paper. These scripts train video classification models (in particular, of the TimeSformer, ViViT, and VideoMAE architectures) for the task of human action recognition.

## Citation

```bibtex
TBD
```

## Remarks & Updates

- (**TBD Date**) The pre-print is released on arXiv.
