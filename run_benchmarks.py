import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader

from data_preprocessors.brats_preprocessor import BraTSPreprocessor
from data_preprocessors.scannet_preprocessor import ScanNetCropPreprocessor, ScanNetIsolatedPreprocessor
from data_preprocessors.shapenet_preprocessor import ShapeNetPairsPreprocessor, ShapeNetBinaryPreprocessor
from dataset import CustomDataset
from model import CNN3DModel
from scripts.evaluate import compute_metrics
from scripts.train import train_model, evaluate_model
from torch_saliency_methods import (GradCAM, GradCAMPlusPlus, HiResCAM, RespondCAM, SaliencyTubes)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global Constants
N_CLASSES = 2

# Training Constants
HIDDEN_DIM = 128
assert (HIDDEN_DIM % 4) == 0
FC_DIM = 512
BATCH_SIZE = 1

# For GPUs with lower memory, lower this so that large samples are excluded
SIZE_THRESH = (0.85) * 1e8

# WSOL and WSSS Constants
EVAL_CONFIG = {
    'CAM_METHODS': [GradCAM, HiResCAM, GradCAMPlusPlus, RespondCAM, SaliencyTubes],
    'N_THRESH_TAU': 100,  # Number of thresholds tau to choose between 0 and 1
    'DELTA_ACC_THRESH_V1': [50],
    'DELTA_ACC_THRESH_V2': [30, 50, 70],
    'N_SAMPLES': 100  # Number of samples over which to compute the MaxBoxAcc from the test set
}


def run_benchmark(config):
    print(
        f"Running benchmark on dataset: {config['dataset']} for saliency methods: {EVAL_CONFIG['CAM_METHODS']}")
    print(f"Outputting to {config['output_path']}")

    ages = [1]
    if config['dataset'] == "BraTS":
        preprocessor = BraTSPreprocessor(config['datasets_base_path'])
        ages = [20]
    elif config['dataset'] == "shapenet-pairs":
        preprocessor = ShapeNetPairsPreprocessor(config['datasets_base_path'])
    elif config['dataset'] == "shapenet-binary":
        preprocessor = ShapeNetBinaryPreprocessor(config['datasets_base_path'])
    elif config['dataset'] == "scannet-isolated":
        preprocessor = ScanNetIsolatedPreprocessor(config['datasets_base_path'])
    elif config['dataset'] == "scannet-crop":
        preprocessor = ScanNetCropPreprocessor(config['datasets_base_path'])
    else:
        # ages = [20, 10]
        raise NotImplementedError(
            f"Invalid dataset {config['dataset']}. "
            f"Please choose one of [BraTS, shapenet-pairs, shapenet-binary, scannet-isolated, scannet-crop]")

    data_dict = preprocessor.preprocess()
    custom_dataset = CustomDataset(data_dict, config['n_folds'], config['fold'])
    train_loader = DataLoader(dataset=custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    assert config['fold'] < config['n_folds']
    model = CNN3DModel(in_channels=preprocessor.n_channels, n_classes=N_CLASSES, hidden_dim=HIDDEN_DIM).cuda()
    train_model(ages=ages, config=config, model=model, n_classes=N_CLASSES, train_loader=train_loader,
                hidden_dim=HIDDEN_DIM)
    evaluate_model(config['fold'], config['n_folds'], data_dict, model)

    compute_metrics(bench_config=config, eval_config=EVAL_CONFIG, model=model, pred_layer=model.fc2,
                    target_layer=model.conv3, data_dict=data_dict)


def main():
    parser = argparse.ArgumentParser(
        prog='SE3D',
        description='Runs the SE3D benchmark')

    parser.add_argument(
        '-d', '--dataset',
        help="The dataset to compute the benchmark on. One of [shapenet-binary, shapenet-pairs, scannet-isolated, "
             "scannet-crop, BraTS]",
        required=True)
    parser.add_argument(
        '-o', '--output_path',
        help="The path where to save the output file. Defaults to 'cwd/data/output.txt'",
        default=Path(os.getcwd()) / "data" / "output.txt")
    parser.add_argument(
        '--datasets_base_path',
        help="The path where datasets are stored. Refer to the guide for help on how to structure the "
             "folder.",
        default=Path(os.getcwd()) / "datasets")
    parser.add_argument(
        '--models_base_path',
        help="The path where models are stored. Refer to the guide for help on how to structure the "
             "folder.",
        default=Path(os.getcwd()) / "models")
    parser.add_argument(
        '--n_folds',
        help="The number of folds to use for cross-validation. Defaults to 5.",
        default=5)
    parser.add_argument(
        '--fold',
        help="The fold to use for cross-validation. Defaults to 0.",
        default=0)
    parser.add_argument(
        '--train',
        action="store_true",
        help="Whether to retrain the models. Defaults to False.",
        default=False)
    # argument to choose a model based on the dataset it was trained on
    parser.add_argument(
        '--model_dataset',
        help="The dataset the model was trained on. One of [shapenet-binary, shapenet-pairs, scannet-isolated, "
             "scannet-crop, BraTS]. Defaults to the same as the dataset.",
        default=None)

    args = vars(parser.parse_args())
    output_path = Path(args["output_path"])
    os.makedirs(output_path.parent, exist_ok=True)
    models_base_path = Path(args["models_base_path"])
    os.makedirs(models_base_path, exist_ok=True)

    benchmark_config = {
        "dataset": args["dataset"],
        "output_path": output_path,
        "datasets_base_path": args["datasets_base_path"],
        "models_base_path": models_base_path,
        "n_folds": int(args["n_folds"]),
        "fold": int(args["fold"]),
        "train": bool(args["train"]),
        "model_dataset": args["model_dataset"] if args["model_dataset"] else args["dataset"]
    }

    run_benchmark(benchmark_config)


if __name__ == "__main__":
    main()
