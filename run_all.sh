# Description: Run all scripts in order
# Usage: bash run_all.sh --metadata_root <path> --train_data_root <path> --data_root <path> --model_path <path> --wandb <bool>

# Pre-requisites:
# You have to manually download the dataset from https://www.med.upenn.edu/cbica/brats2020/data.html

# Parameters:
#   --metadata_root: path to metadata root
#   --train_data_root: path to BraTS 2020 training data root (e.g. /home/brats2020/MICCAI_BraTS2020_TrainingData)
#   --data_root: path to prepared data root
#   --model_path: path to model root
#   --wandb: whether to use wandb

# Default Values
metadata_root="."
train_data_root="data/MICCAI_BraTS2020_TrainingData"
data_root="data"
model_path="3D_CNN"
wandb="False"

# Parse Arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --metadata_root)
    metadata_root="$2"
    shift
    shift
    ;;
  --train_data_root)
    train_data_root="$2"
    shift
    shift
    ;;
  --data_root)
    data_root="$2"
    shift
    shift
    ;;
  --model_path)
    model_path="$2"
    shift
    shift
    ;;
  --wandb)
    wandb="$2"
    shift
    shift
    ;;
  *)
    shift
    ;;
  esac
done

# If wandb is True, then add --wandb, else remove it
if [ "$wandb" = "True" ]; then
  wandb="--wandb True"
else
  wandb=""
fi

# Prepare Dataset
python prepare_dataset.py --train_data_root $train_data_root --output_dir $data_root

# Train and Evaluate on All 5 Folds
for fold in {0..4}; do
  python train.py --fold $fold --metadata_root $metadata_root --model_path $model_path $wandb
  python evaluate.py --fold $fold --metadata_root $metadata_root --model_path $model_path
done

# Generate Results
python results_to_md.py --metadata_root $metadata_root --model_path $model_path
