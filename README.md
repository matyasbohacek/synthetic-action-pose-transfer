![banner](https://github.com/user-attachments/assets/1e6b6bd1-93ba-4658-b014-b9ab43c6badb)

# Synthetic Human Action Video Data Generation with Pose Transfer

#### [Vaclav Knapp](https://vaclavknapp.github.io) and [Maty Bohacek](https://www.matyasbohacek.com)

In video understanding tasks, particularly those involving human motion, synthetic data generation often suffers from uncanny features, diminishing its effectiveness for training. Tasks such as sign language translation, gesture recognition, and human motion understanding in autonomous driving have thus been unable to exploit the full potential of synthetic data. This paper proposes a method for generating synthetic human action video data using pose transfer (specifically, controllable 3D Gaussian avatar models). We evaluate this method on the Toyota Smarthome and NTU RGB+D datasets and show that it improves performance in action recognition tasks. Moreover, we demonstrate that the method can effectively scale few-shot datasets, making up for groups underrepresented in the real training data and adding diverse backgrounds. We open-source the method along with RANDOM People, a dataset with videos and avatars of novel human identities for pose transfer crowd-sourced from the internet.

> [Website](https://synthetic-human-action.github.io) — [Paper](https://openreview.net/pdf?id=KTXL0idiky) — [Data](https://huggingface.co/datasets/matybohacek/RANDOM-People) — [Poster]() — [Contact us](mailto:maty-at-stanford-dot-edu)
> 
> _Synthetic Data for Computer Vision Workshop @ CVPR 2025_

## Getting Started

TBD

## Data

If you want to use our _RANDOM People_ dataset, refer to [our Hugging Face page](https://huggingface.co/datasets/matybohacek/RANDOM-People). If you wish to reproduce our results on the Toyota Smarthome dataset, please refer to [the official distribution instructions](https://project.inria.fr/toyotasmarthome/), as presented by its authors.

## Features

[**Synthetic Data Generation.**](synthetic_data_generation/) Synthetic video data generation toolkit built around pose transfer. It can be used to scale an arbitrary dataset with videos of human action.

[**Experiments.**](experiments/) This modules includes scripts for baseline and few-shot experiments described in the paper. These scripts train video classification models for the task of human action recognition.

## Citation

```bibtex
@inproceedings{knapp2025synthetic,
  title={Synthetic Human Action Video Data Generation with Pose Transfer},
  author={Knapp, Vaclav and Bohacek, Matyas},
  booktitle={Synthetic Data for Computer Vision Workshop @ CVPR 2025}
}
```

## Remarks & Updates

- (**June 2025**) The paper is presented at a CVPR workshop in Nashville, TN.
