# Synthetic Data Generation

This repository provides tools for generating synthetic data using the ExAvatar framework. Follow the steps below to set up the environment and run the fitting process for avatars and motion videos.

---

## Prerequisites

1. **Python**: Ensure Python 3.10 is installed.
2. **Dependencies**: Install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

---

## Fitting Avatars

### Using the RANDOM PEOPLE Dataset

1. Clone the ExAvatar repository:
   ```bash
   cd Pose-Transfer-synthetic-data
   git clone https://github.com/mks0601/ExAvatar_RELEASE.git
   ```
   Follow the dowloading instruction for models in `ExAvatar_RELEASE` repository
2. Navigate to the dataset folder:
   ```bash
   cd ./ExAvatar_RELEASE/fitting/data/Custom/data
   ```
3. Download the RANDOM PEOPLE dataset and place `avatars` folder in this `.` directory.
4. Place the `avatars_checkpoints` folder in `Pose-Transfer-synthetic-data/ExAvatar_RELEASE/avatar/output/model_dump`
   ```bash
   mv ./ExAvatar_RELEASE/fitting/data/Custom/data/RANDOM_people/avatars_checkpoints Pose-Transfer-synthetic-data/ExAvatar_RELEASE/avatar/output/model_dump
   cd Pose-Transfer-synthetic-data/ExAvatar_RELEASE/avatar/output/model_dump
   mv avatars_checkpoints avatars
   ```

### Using Your Own Avatars

1. To use your own monocular videos, place them in:
   ```
   mkdir -p ./ExAvatar_RELEASE/fitting/data/Custom/data/avatars
   cd ./ExAvatar_RELEASE/fitting/data/Custom/data/avatars
   ```
   Ensure the data follows the same directory structure as the monocular videos from RANDOM PEOPLE dataset.

2. Replace the default `run.py` file and add additional files:
   ```bash
   rm ./ExAvatar_RELEASE/fitting/tools/run.py
   mv Pose-Transfer-synthetic-data/synthetic_data_generation/run.py Pose-Transfer-synthetic-data/ExAvatar_RELEASE/fitting/tools/
   mv Pose-Transfer-synthetic-data/synthetic_data_generation/normalise_vid.py Pose-Transfer-synthetic-data/ExAvatar_RELEASE/fitting/tools/
   mv Pose-Transfer-synthetic-data/synthetic_data_generation/n_frames.py Pose-Transfer-synthetic-data/ExAvatar_RELEASE/fitting/tools/
   mv Pose-Transfer-synthetic-data/synthetic_data_generation/extract_frames.py Pose-Transfer-synthetic-data/ExAvatar_RELEASE/fitting/tools/
   ```

3. Navigate to the `tools` folder:
   ```bash
   cd ./ExAvatar_RELEASE/fitting/tools
   ```

4. Run the fitting script:
   ```bash
   python run.py --rootpath path/to/your/video_folder
   ```
5. Training the 3D avatars:
   ```bash
   python train.py --subject_id path/to/your/video_folder
   ```
   The checkpoints are saved in `Pose-Transfer-synthetic-data/ExAvatar_RELEASE/avatar/output/model_dump`
   Place the folder with checkpoints inside `avatars` folder
---

## Fitting Motion Videos

1. Place your dataset folder containing `.mp4` videos in:
   ```
   ./ExAvatar_RELEASE/fitting/data/Custom/data/
   ```

2. Modify paths in the `mp4.sh` script to point to your dataset, then run:
   ```bash
   bash mp4.sh
   ```

3. Add motion fitting script:
   ```bash
   mv Pose-Transfer-synthetic-data/synthetic_data_generation/motion_run.py Pose-Transfer-synthetic-data/ExAvatar_RELEASE/fitting/tools/
   ```

4. Navigate to the `tools` folder:
   ```bash
   cd ./ExAvatar_RELEASE/fitting/tools
   ```

5. Run the motion fitting script:
   ```bash
   bash motion_loop.sh
   ```
   Make sure to add path to the dataset folder
## Animating videos

1. Replace old animating files with new ones:
   ```bash
   rm Pose-Transfer-synthetic-data/ExAvatar_RELEASE/main/animate.py
   rm Pose-Transfer-synthetic-data/ExAvatar_RELEASE/avatar/common/nets/module.py
   mv Pose-Transfer-synthetic-data/synthetic_data_generation/animate.py Pose-Transfer-synthetic-data/ExAvatar_RELEASE/main
   mv Pose-Transfer-synthetic-data/synthetic_data_generation/module.py Pose-Transfer-synthetic-data/ExAvatar_RELEASE/avatar/common/nets
   mv Pose-Transfer-synthetic-data/synthetic_data_generation/animate_all_data.sh Pose-Transfer-synthetic-data/ExAvatar_RELEASE/main
   ```
2. Add background images in `.png` format to the `Pose-Transfer-synthetic-data/ExAvatar_RELEASE/images`
3. Animate all videos in you video dataset with all avatars and images:
   ```bash
   cd Pose-Transfer-synthetic-data/ExAvatar_RELEASE/main
   bash animate_all_data.sh 
   ```
   Make sure to check correct paths in the `animate_all_data.sh`
   
