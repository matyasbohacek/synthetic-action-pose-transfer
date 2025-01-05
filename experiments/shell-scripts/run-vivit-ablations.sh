python train.py \
    --baseline_model "google/vivit-b-16x2-kinetics400" \
    --training_sources "real" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "vivit-ablations-real" \
    --batch_size 2

python train.py \
    --baseline_model "google/vivit-b-16x2-kinetics400" \
    --training_sources "white" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "vivit-ablations-white" \
    --batch_size 2

python train.py \
    --baseline_model "google/vivit-b-16x2-kinetics400" \
    --training_sources "background" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "vivit-ablations-background" \
    --batch_size 2

python train.py \
    --baseline_model "google/vivit-b-16x2-kinetics400" \
    --training_sources "real" "white" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "vivit-ablations-real-white" \
    --batch_size 2

python train.py \
    --baseline_model "google/vivit-b-16x2-kinetics400" \
    --training_sources "white" "background" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "vivit-ablations-white-background" \
    --batch_size 2

python train.py \
    --baseline_model "google/vivit-b-16x2-kinetics400" \
    --training_sources "real" "background" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "vivit-ablations-real-background" \
    --batch_size 2

python train.py \
    --baseline_model "google/vivit-b-16x2-kinetics400" \
    --training_sources "real" "white" "background" \
    --class_cap_per_dir__real 75 \
    --class_cap_per_dir__background 75 \
    --class_cap_per_dir__white 75 \
    --experiment_name "vivit-ablations-real-white-background" \
    --batch_size 2