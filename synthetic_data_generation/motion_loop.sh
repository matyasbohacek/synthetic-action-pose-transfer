#!/bin/bash

# dataset path
DATA_DIR="/workspace/Maty_CVPR/ExAvatar_RELEASE/fitting/data/Custom/data/wacv_motion"


MOTION_RUN_SCRIPT="motion_run.py"


for folder_path in "$DATA_DIR"/*/; do
    echo "Processing folder: $folder_path"


    python "$MOTION_RUN_SCRIPT" --root_path "$folder_path"

    if [ $? -ne 0 ]; then
        echo "An error occurred while processing $folder_path. Exiting."
        exit 1
    fi

done

echo "All folders have been processed."
