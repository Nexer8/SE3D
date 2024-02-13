# Description: Run all scripts in order
# Usage: bash run_all.sh --metadata_root <path> --train_data_root <path> --datasets_root <path> --model_root <path>

# Pre-requisites:
# You have to manually download the dataset from https://www.med.upenn.edu/cbica/brats2020/data.html

# Parameters:
#   --metadata_root: path to metadata root
#   --train_data_root: path to BraTS 2020 training data root (e.g. /home/brats2020/MICCAI_BraTS2020_TrainingData)
#   --datasets_root: path to prepared datasets root
#   --model_root: path to model root

# Default Values
metadata_root="."
train_data_root="BraTS/MICCAI_BraTS2020_TrainingData"
datasets_root="datasets"
model_root="models"

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
  --datasets_root)
    datasets_root="$2"
    shift
    shift
    ;;
  --model_root)
    model_root="$2"
    shift
    shift
    ;;
  *)
    shift
    ;;
  esac
done

# Prepare Dataset (for BraTS 2020 only)
python scripts/prepare_dataset.py \
    --train_data_root $train_data_root \
    --output_dir $datasets_root

# Train and Evaluate BraTS on All 5 Folds
for fold in {0..4}; do
    python run_benchmarks.py \
        -d BraTS \
        -o data/output_BraTS_fold$fold.txt \
        --datasets_base_path $datasets_root \
        --fold $fold \
        --train \
        --models_base_path $model_root
done

# Train and Evaluate shapenet-binary on 1 fold
python run_benchmarks.py \
    -d shapenet-binary \
    -o data/output_shapenet-binary_fold0.txt \
    --datasets_base_path $datasets_root \
    --fold 0 \
    --train \
    --models_base_path $model_root

# Train and Evaluate shapenet-pairs on 1 fold
python run_benchmarks.py \
    -d shapenet-pairs \
    -o data/output_shapenet-pairs_fold0.txt \
    --datasets_base_path $datasets_root \
    --fold 0 \
    --train \
    --models_base_path $model_root

# Train and Evaluate scannet-isolated on 1 fold
python run_benchmarks.py \
    -d scannet-isolated \
    -o data/output_scannet-isolated_fold0.txt \
    --datasets_base_path $datasets_root \
    --fold 0 \
    --train \
    --models_base_path $model_root

# Train and Evaluate scannet-crop on 1 fold
python run_benchmarks.py \
    -d shapenet-binary \
    -o data/output_scannet-crop_fold0.txt \
    --datasets_base_path $datasets_root \
    --fold 0 \
    --train \
    --models_base_path $model_root
