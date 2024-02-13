# *SE3D*: A Framework for Saliency Method Evaluation in 3D Medical Imaging

**WORK IN PROGRESS. Refined documentation will be available in the upcoming days.**

Simple launch of the experiments:

```sh
python run_benchmarks.py -d <BraTS|shapenet-pairs|shapenet-binary|scannet-isolated|scannet-crop>
```

Authors: `REDACTED` *(available upon publication)*

> Deep learning models are seeing widespread use in the medical field, with Convolutional Neural Networks (CNNs) being
> used in scenarios such as histopathology and X-ray imaging. More recently, 3D CNNs have also been employed to analyze
> volumetric data from various medical imaging modalities such as MRI and CT scans.
> Despite advances in Explainable Artificial Intelligence, little effort has been put into constructing saliency maps
> for 3D CNNs. Consequently, the use of such models in critical scenarios is hindered by their black-box,
> non-interpretable nature. To address this issue, we propose SE3D: a framework for saliency method evaluation in 3D
> medical imaging, based on a new benchmark built upon the BraTS 2020 brain tumor segmentation dataset. We also extend
> popular 2D saliency methods to 3D data, achieving performances that match those of saliency methods designed for 3D
> CNNs. Although 2D saliency methods have been widely used as means of model explanation, our results suggest that there
> is margin for future improvements, to enable safer applications of 3D CNNs in the medical field.

## Table of Contents

- [Dataset and Preprocessing](#dataset-and-preprocessing)
    - [Downloading the Dataset](#downloading-the-dataset)
    - [Preprocessing](#preprocessing)
- [Saliency Methods](#saliency-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Reproducing the Results](#reproducing-the-results)
    - [Requirements](#requirements)
    - [Code Structure](#code-structure)
    - [Running the Experiments](#running-the-experiments)
- [Code License](#code-license)

## Dataset and Preprocessing

The paper demonstrates the use of the proposed framework on the BraTS 2020 brain tumor segmentation dataset. This
however does not mean that the framework is limited to this dataset, as it can be used with any dataset containing 3D
volumes and corresponding segmentation masks.

### Downloading the Dataset

The BraTS 2020 dataset can be downloaded from
the [official website](https://www.med.upenn.edu/cbica/brats2020/data.html). The dataset is split into two parts:
training and validation. The training set contains 369 volumes, while the validation set contains 125 volumes. The
dataset is provided in the form of NIfTI files, which can be opened using the [NiBabel](https://nipy.org/nibabel/)
library.

### Preprocessing

The scans are provided in the form of 4D volumes, with the fourth dimension representing the different MRI modalities.
The modalities are:

- native (**T1**)
- post-contrast T1-weighted (**T1Gd**),
- T2-weighted (**T2**),
- T2 Fluid Attenuated Inversion Recovery (**T2-FLAIR**).

The ground truth segmentation masks are provided in the form of 3D volumes, with each voxel containing a label from the
following set:

- 0: background,
- 1: necrotic and non-enhancing tumor core (**NCR/NET**),
- 2: peritumoral edema (**ED**),
- 4: GD-enhancing tumor (**ET**).

The preprocessing steps are implemented in the `prepare_dataset.py` script. The script takes as input the path to the
training folder of the BraTS 2020 dataset, and creates a folder containing the preprocessed dataset.

The following preprocessing steps are performed:

1. Each volume is rotated by $-90^{\circ}$ around $x$ and $y$ axes. This is done to align the volumes with the standard
   radiological view, where the left side of the brain is on the right side of the image.
2. Each volume is first min-max normalized to the range $[0, 1]$ and then converted to the range $[0, 255]$. This is
   done to reduce the memory footprint of the dataset, as the original volumes are stored as 32-bit floating point
   numbers.
3. The segmentation masks are converted to contain only two labels: background (0) and tumor (1). This is done by
   merging the labels 1, 2 and 4 into a single label. The intuition behind this is that the tumor is the only region of
   interest for the saliency methods, and the different labels are not relevant for the evaluation.
4. Both volumes and segmentation masks are split into 2 halves along the *corpus callosum* plane. Thanks to this, the
   resulting dataset contains both volumes with and without the tumor.

To run the preprocessing script, use the following command:

```sh
python prepare_dataset.py --train_data_root <path_to_train_data_root> \
                            --output_dir <path_to_output_dir>
```

## Saliency Methods

All saliency methods are implemented as Keras callbacks, and can be used with any Keras model for input data of any
dimensionality. The methods are implemented in the `tf_saliency_methods` module.

Available methods:

- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- [Grad-CAM++](https://arxiv.org/abs/1710.11063)
- [HiResCAM](https://arxiv.org/abs/2011.08891)
- [Respondd-CAM](https://arxiv.org/abs/1806.00102)
- [Saliency Tubes](https://arxiv.org/abs/1902.01078)

See module's `README.md` for more details.

## Evaluation Metrics

The evaluation metrics are implemented in the `wsol_3d_metrics` module. The following metrics are available:

- `MaxBoxAcc`: Maximal Box Accuracy, introduced in the [paper](https://ieeexplore.ieee.org/document/9762560). It is a 2D
  binary metric, which computes the *Intersection over Union (IoU)* between the bounding box over the largest connected
  component of the prediction heatmap and the ground truth over various heatmap thresholds $\tau_1, \tau_2, \dots,
  \tau_l$ and assigns `1` if for at least one threshold $\tau_i$ the *IoU* is greater than a threshold $\delta$ and `0`
  otherwise.
- `MaxBoxAccV2`: Maximal Box Accuracy V2, which is a variant of `MaxBoxAcc` introduced in the same paper. The difference
  to `MaxBoxAcc` is that it allows for multiple contours in the ground truth and prediction. It is a 2D metric, which
  computes the *IoU* between bounding boxes around all the connected components of the prediction heatmap and the ground
  truth over various heatmap thresholds $\tau_1, \tau_2, \dots, \tau_l$ and assigns `1` if for at least one threshold
  $\tau_i$ there is at least one pair of bounding boxes with *IoU* greater than a threshold $\delta$ and `0` otherwise.
- `PxAP`: Pixel Average Precision, introduced in the same paper. It is computed as the area under the pixel-wise
  precision-recall curve of the prediction heatmap with respect to the ground truth segmentation mask over various
  heatmap thresholds $\tau_1, \tau_2, \dots, \tau_l$.
- `Max3DBoxAcc`: Maximal 3D Box Accuracy, introduced in *"SE3D: A Framework for Saliency Method Evaluation in 3D Medical
  Imaging"*. It is a 3D extension of `MaxBoxAcc`.
- `Max3DBoxAccV2`: Maximal 3D Box Accuracy V2, introduced in *SE3D*. It is a 3D extension of `MaxBoxAccV2`.
- `VxAP`: Voxel Average Precision, introduced in *SE3D*. It is a 3D extension of `PxAP`.

Check the module's `README.md` for more details.

## Results

Model's accuracy evaluated on all 5 folds: $76.16\% \pm 0.57\%$ (*mean $\pm$ standard error*).

| Method         | MaxBoxAcc (IoU: 0.5) | MaxBoxAcc V2 (IoU: [0.3, 0.5, 0.7])                           | VxAP               | Max3DBoxAcc (IoU: 0.5) | Max3DBoxAcc V2 (IoU: [0.3, 0.5, 0.7])                          |
|----------------|----------------------|---------------------------------------------------------------|--------------------|------------------------|----------------------------------------------------------------|
| Grad-CAM       | 5.26 &plusmn; 0.62   | [14.81 &plusmn; 0.92, 5.75 &plusmn; 0.55, 1.14 &plusmn; 0.12] | 6.92 &plusmn; 1.04 | 16.17 &plusmn; 2.82    | [47.48 &plusmn; 4.09, 16.87 &plusmn; 2.77, 3.30 &plusmn; 1.08] | 
| HiResCAM       | 2.25 &plusmn; 0.21   | [8.65 &plusmn; 0.47, 2.69 &plusmn; 0.18, 0.45 &plusmn; 0.04]  | 4.45 &plusmn; 0.55 | 10.09 &plusmn; 1.70    | [39.30 &plusmn; 2.97, 10.61 &plusmn; 1.75, 1.22 &plusmn; 0.19] | 
| Grad-CAM++     | 6.32 &plusmn; 0.91   | [16.05 &plusmn; 1.18, 6.87 &plusmn; 0.78, 1.37 &plusmn; 0.15] | 9.70 &plusmn; 2.13 | 18.09 &plusmn; 2.78    | [49.22 &plusmn; 4.88, 18.61 &plusmn; 2.98, 3.83 &plusmn; 1.36] | 
| Saliency Tubes | 5.63 &plusmn; 1.54   | [13.60 &plusmn; 1.84, 6.84 &plusmn; 1.67, 1.96 &plusmn; 0.60] | 8.68 &plusmn; 2.98 | 14.78 &plusmn; 4.14    | [41.22 &plusmn; 4.16, 17.04 &plusmn; 4.02, 2.26 &plusmn; 0.72] | 

## Reproducing the Results

The code provided in this repository allows to reproduce the results presented in the paper. The steps to do so are
described below.

### Requirements

To check the requirements, see the `requirements.txt` file. To install the requirements, simply run the following
command:

```sh
pip install -r requirements.txt
```

### Code Structure

The code is structured as follows:

```sh
SE3D                    # root directory
├── 3D_CNN              # 3D CNN model weights and training logs for all 5 folds
├── README.md           # this file
├── evaluate.py         # script for evaluating the saliency methods
├── metadata.csv        # metadata file for the customized BraTS 2020 dataset
├── model.py            # 3D CNN model architecture definition
├── prepare_dataset.py  # script for creating the customized BraTS 2020 dataset
├── requirements.txt    # requirements file
├── results_to_md.py    # script for converting the results to Markdown format
├── run_all.sh          # script for running all experiments
├── tf_saliency_methods # module implementing the saliency methods
├── train.py            # script for training the 3D CNN model
└── wsol_3d_metrics     # module implementing the evaluation metrics
```

### Running the Experiments

To reproduce the experiments, execute the following shell script:

```sh
./run_all.sh --metadata_root <path_to_metadata_root> \
                --train_data_root <path_to_BraTS2020_TrainingData> \
                --data_root <output_path_for_customized_BraTS2020_dataset> \
                --model_path <output_path_for_model_weights> \
                --wandb <flag_for_logging_to_wandb>
```

The script will perform the following steps:

1. Create the customized BraTS 2020 dataset.
2. Train the 3D CNN model on the customized BraTS 2020 dataset on all 5 folds.
3. Evaluate the saliency methods on the trained model on all 5 folds.
4. Convert the results to *Markdown* format.

To manually reproduce the experiments, see the `run_all.sh` script for the commands to run. In case of running the
experiments with `--wandb` flag set, make sure to set the `WANDB_API_KEY`, `WANDB_BRATS_PROJECT`, and `WANDB_ENTITY`
environment variables to the appropriate values.

## Acknowledgements

This code is based on the [wsolevaluation](https://github.com/clovaai/wsolevaluation) work.

## Code License

To be determined.
