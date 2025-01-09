#!/bin/bash

# Source directory containing the dataset with .mp4 videos
SRC_DIR=""

# Target directory where videos will be copied
TARGET_DIR="Pose-transfer-syntetic-data/ExAvatar_RELEASE/fitting/data/Custom/data/motion"

mkdir -p "$TARGET_DIR" 

for video_path in "$SRC_DIR"/*.mp4; do

  video_name=$(basename "$video_path")
  dir_name="${video_name%.mp4}"


  mkdir -p "$TARGET_DIR/$dir_name"


  cp -r "$video_path" "$TARGET_DIR/$dir_name/video.mp4"
done
