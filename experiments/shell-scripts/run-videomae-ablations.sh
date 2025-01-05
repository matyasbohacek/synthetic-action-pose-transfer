python train.py \
    --baseline_model "MCG-NJU/videomae-base" \
    --training_sources "real" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "ablations-real"

python train.py \
    --baseline_model "MCG-NJU/videomae-base" \
    --training_sources "white" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "ablations-white"

python train.py \
    --baseline_model "MCG-NJU/videomae-base" \
    --training_sources "background" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "ablations-background"

python train.py \
    --baseline_model "MCG-NJU/videomae-base" \
    --training_sources "real" "white" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "ablations-real-white"

python train.py \
    --baseline_model "MCG-NJU/videomae-base" \
    --training_sources "white" "background" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "ablations-white-background"

python train.py \
    --baseline_model "MCG-NJU/videomae-base" \
    --training_sources "real" "background" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "ablations-real-background"

python train.py \
    --baseline_model "MCG-NJU/videomae-base" \
    --training_sources "real" "white" "background" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "ablations-real-white-background"