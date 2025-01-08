#!/bin/bash

# Directories
SUBJECT_DIR="/workspace/Maty_CVPR/ExAvatar_RELEASE/fitting/data/Custom/data/avatars" #folder with all avatars
MOTION_DIR="/workspace/Maty_CVPR/ExAvatar_RELEASE/fitting/data/Custom/data/motion" #folder with all motion videos
IMAGES_DIR="/workspace/Maty_CVPR/ExAvatar_RELEASE/images" #folder with images for backgrounds


SUBJECT_IDS=($(find "$SUBJECT_DIR" -mindepth 1 -maxdepth 1 -type d ! -name "motion" -exec basename {} \;))


MOTION_PATHS=($(find "$MOTION_DIR" -mindepth 1 -maxdepth 1 -type d))


IMAGES=($(find "$IMAGES_DIR" -type f))


for SUBJECT_ID in "${SUBJECT_IDS[@]}"; do

    for MOTION_PATH in "${MOTION_PATHS[@]}"; do

        RANDOM_IMAGES=($(printf "%s\n" "${IMAGES[@]}" | shuf -n 3))

        for IMAGE_PATH in "${RANDOM_IMAGES[@]}"; do

            FULL_SUBJECT_PATH="avatars/$SUBJECT_ID"

            CMD="python animate.py --subject_id \"$FULL_SUBJECT_PATH\" --test_epoch 4 --motion_path \"$MOTION_PATH\" --image_path \"$IMAGE_PATH\""
            echo "Running: $CMD"
            python animate.py --subject_id "$FULL_SUBJECT_PATH" --test_epoch 4 --motion_path "$MOTION_PATH" --image_path "$IMAGE_PATH"
        done
    done
done
