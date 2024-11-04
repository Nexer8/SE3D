# *SE3D*: A Framework for Saliency Method Evaluation in 3D Imaging

Authors: [*Mariusz Wiśniewski*](https://linkedin.com/in/mariusz-krzysztof-wisniewski/), [*Loris Giulivi*](https://linkedin.com/in/lorisgiulivi/), [*Giacomo Boracchi*](https://linkedin.com/in/giacomoboracchi/).

> For more than a decade, deep learning models have been dominating in various 2D imaging tasks. Their application is now extending to 3D imaging, with 3D Convolutional Neural Networks (3D CNNs) being able to process LIDAR, MRI, and CT scans, with significant implications for fields such as autonomous driving and medical imaging. In these critical settings, explaining the model's decisions is fundamental. Despite recent advances in Explainable Artificial Intelligence, however, little effort has been devoted to explaining 3D CNNs, and many works explain these models via inadequate extensions of 2D saliency methods.
>
> One fundamental limitation to the development of 3D saliency methods is the lack of a benchmark to quantitatively assess them on 3D data. To address this issue, we propose SE3D: a framework for **S**aliency method **E**valuation in **3D** imaging. We propose modifications to ShapeNet, ScanNet, and BraTS datasets, and evaluation metrics to assess saliency methods for 3D CNNs. We evaluate both state-of-the-art saliency methods designed for 3D data and extensions of popular 2D saliency methods to 3D. Our experiments show that 3D saliency methods do not provide explanations of sufficient quality, and that there is margin for future improvements and safer applications of 3D CNNs in critical fields.

For more details, please refer to our publication: [SE3D: A Framework for Saliency Method Evaluation in 3D Imaging](https://doi.org/10.1109/ICIP51287.2024.10647305).

## Table of Contents

- [Datasets and Preprocessing](#datasets-and-preprocessing)
  - [Downloading the Datasets](#downloading-the-datasets)
  - [Preprocessing](#preprocessing)
- [Saliency Methods](#saliency-methods)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Reproducing the Results](#reproducing-the-results)
  - [Requirements](#requirements)
  - [Code Structure](#code-structure)
  - [Running the Experiments](#running-the-experiments)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Code License](#code-license)

## Datasets and Preprocessing

The paper demonstrates the use of the proposed framework on [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/), [ScanNet](http://www.scan-net.org), and [ShapeNet](https://shapenet.org) datasets. It also provides code for running the experiments on [KiTS19](https://kits19.grand-challenge.org) and [Medical Decathlon's Lung](http://medicaldecathlon.com) datasets. The authors, however, did not manage to obtain satisfactory results on the latter two datasets, and thus the corresponding models, as well as results, are not included in this repository. This however does not mean that the framework is limited to these datasets, as it can be used with any dataset containing 3D volumes and corresponding segmentation masks.

### Downloading the Datasets

- The BraTS 2020 dataset can be downloaded from the [official website](https://www.med.upenn.edu/cbica/brats2020/data.html). The dataset is provided in the form of NIfTI files, which can be opened using the [NiBabel](https://nipy.org/nibabel/) library.
- The instructions how to ScanNet's data are available in the [official GitHub repository](https://github.com/ScanNet/ScanNet). The files are provided in the `.ply` format, and can be opened using the [Open3D](http://www.open3d.org) library.
- ShapeNet's data can be downloaded from the [official website](https://shapenet.org). The dataset is provided in the form of 3D models in the `.binvox` format, which can be opened using the [binvox](https://www.patrickmin.com/binvox/) tool.
- Medical Decathlon's Lung dataset can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) or from [AWS](http://medicaldecathlon.com/dataaws/). Similarly to Brats 2020, the dataset contains NIfTI files.
- KiTS19 dataset can be downloaded from the [official GitHub repository](https://github.com/neheller/kits19). The dataset consists of NIfTI files.

### Preprocessing

#### BraTS 2020

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

1. Each volume is rotated by $-90^{\circ}$ around $x$ and $y$ axes. This is done to align the volumes with the standard radiological view, where the left side of the brain is on the right side of the image.
2. Each volume is first min-max normalized to the range $[0, 1]$ and then converted to the range $[0, 255]$. This is done to reduce the memory footprint of the dataset, as the original volumes are stored as 32-bit floating point numbers.
3. The segmentation masks are converted to contain only two labels: background (0) and tumor (1). This is done by merging the labels 1, 2 and 4 into a single label. The intuition behind this is that the tumor is the only region of interest for the saliency methods, and the different labels are not relevant for the evaluation.
4. Both volumes and segmentation masks are split into 2 halves along the *corpus callosum* plane. Thanks to this, the resulting dataset contains both volumes with and without the tumor.

To run the preprocessing script, use the following command:

```sh
python prepare_dataset.py --train_data_root <path_to_train_data_root> \
                            --output_dir <path_to_output_dir>
```

***Note**: The preprocessing script is designed to work with the BraTS 2020 dataset. It can be easily adapted to work with KiTS19, Medical Decathlon's Lung, and other datasets.*

#### ScanNet

The volumes are extracted in two ways:

- `scannet-isolated`: each point cloud gets filtered and only the points belonging to a particular object instance get extracted.
- `scannet-crop`: each point cloud gets cropped to a bounding box around a particular object instance, possibly containing points pertaining to other objects.

#### ShapeNet

The dataset is used to generate two new datasets:

- `shapenet-binary`: the dataset contains only isolated objects corresponding to two classes: `chair` and `table`. The dataset is used to evaluate the saliency methods on a binary classification task.
- `shapenet-pairs`: the dataset contains pairs of adjoined objects corresponding to two classes: `airplane` and `bench`. The dataset is used to evaluate *mass concentration* of the saliency methods.

## Saliency Methods

All saliency methods are implemented as in both Torch and Keras callbacks, and can be used with any PyTorch and Keras model for input data of any dimensionality. The methods are implemented in `torch_saliency_methods` and `tf_saliency_methods` modules.

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
- `MaxF1`: Maximal F1 Score, introduced in *SE3D*. It is a sample-wise average F1 score between the thresholded saliency maps $\{s_x \ge \tau\}$ and the ground truth segmentation masks $m_x$ computed at the optimal $\tau$.
- `Prec@F1τ`: Precision at Optimal F1, introduced in *SE3D*. It computes the `VxPrec` over the maps $\{s_x \ge \tau_{F1}\}$ thresholded at the optimal $\tau_{F1}$, where $\tau_{F1}$ is the threshold that maximizes the F1 score.
- `Rec@F1τ`: Recall at Optimal F1, introduced in *SE3D*. It computes the `VxRec` over the maps $\{s_x \ge \tau_{F1}\}$ thresholded at the optimal $\tau_{F1}$, where $\tau_{F1}$ is the threshold that maximizes the F1 score.
- `MC`: Mass Concentration, introduced in *SE3D*, is specifically designed for the paired `shapenet-pairs` dataset and evaluates whether the saliency map focuses **only** on the object of interest.

Check the module's `README.md` for more details.

## Results

<table style="text-align:center;">
  <caption>Results of saliency map evaluation for 2D and 3D metrics. Since in our case <code>VxAP</code>=<code>PxAP</code>, we only report the former.</caption>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center;">Dataset <br>&lt;classes&gt; <br>(test accuracy)</th>
      <th rowspan="2" style="text-align:center;">Saliency Method</th>
      <th colspan="2" style="text-align:center;">3D WSOL Metrics</th>
      <th colspan="4" style="text-align:center;">3D WSSS Metrics</th>
      <th colspan="2" style="text-align:center;">2D WSOL Metrics</th>
    </tr>
    <tr>
      <th style="text-align:center;">Max3DBoxAcc <br> (&delta; = 0.5)</th>
      <th style="text-align:center;">Max3DBoxAccV2 <br> (&delta; &in; {0.3, 0.5, 0.7})</th>
      <th style="text-align:center;">VxAP</th>
      <th style="text-align:center;">MaxF1</th>
      <th style="text-align:center;">Prec@F1&tau;</th>
      <th style="text-align:center;">Rec@F1&tau;</th>
      <th style="text-align:center;">MaxBoxAcc <br> (&delta; = 0.5)</th>
      <th style="text-align:center;">MaxBoxAccV2 <br> (&delta; &in; {0.3, 0.5, 0.7})</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" style="text-align:center; width:1%; white-space:nowrap;">
        <div style="transform:rotate(-90deg); display:inline-block;">shapenet-binary <br> chair/table <br> (0.989)</div>
      </td>
      <td>Grad-CAM</td>
      <td>0.38</td>
      <td>0.42</td>
      <td>0.11</td>
      <td>0.18</td>
      <td>0.11</td>
      <td>0.59</td>
      <td>0.14</td>
      <td>0.18</td>
    </tr>
    <tr>
      <td>Grad-CAM++</td>
      <td>0.18</td>
      <td>0.35</td>
      <td>0.10</td>
      <td>0.18</td>
      <td>0.09</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>0.14</td>
    </tr>
    <tr>
      <td>HiResCAM</td>
      <td>0.24</td>
      <td>0.34</td>
      <td>0.10</td>
      <td>0.19</td>
      <td>0.11</td>
      <td>0.60</td>
      <td>0.10</td>
      <td>0.14</td>
    </tr>
    <tr>
      <td>Respond-CAM</td>
      <td>0.11</td>
      <td>0.22</td>
      <td>0.10</td>
      <td>0.18</td>
      <td>0.10</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>0.09</td>
    </tr>
    <tr>
      <td>Saliency Tubes</td>
      <td>0.29</td>
      <td>0.40</td>
      <td>0.23</td>
      <td>0.30</td>
      <td>0.25</td>
      <td>0.38</td>
      <td>0.19</td>
      <td>0.29</td>
    </tr>
    <tr style="border-bottom: 3px;">
      <td colspan="10"></td>
    </tr>
    <tr>
      <td rowspan="5" style="text-align:center; width:1%; white-space:nowrap;">
        <div style="transform:rotate(-90deg); display:inline-block;">scannet-isolated <br> chair/table <br> (0.919)</div>
      </td>
      <td>Grad-CAM</td>
      <td>0.39</td>
      <td>0.41</td>
      <td>0.05</td>
      <td>0.11</td>
      <td>0.07</td>
      <td>0.28</td>
      <td>0.02</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>Grad-CAM++</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.10</td>
      <td>0.06</td>
      <td>0.24</td>
      <td>0.01</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>HiResCAM</td>
      <td>0.00</td>
      <td>0.10</td>
      <td>0.05</td>
      <td>0.12</td>
      <td>0.08</td>
      <td>0.26</td>
      <td>0.01</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>Respond-CAM</td>
      <td>0.01</td>
      <td>0.18</td>
      <td>0.05</td>
      <td>0.11</td>
      <td>0.08</td>
      <td>0.20</td>
      <td>0.02</td>
      <td>0.08</td>
    </tr>
    <tr>
      <td>Saliency Tubes</td>
      <td>0.63</td>
      <td>0.61</td>
      <td>0.09</td>
      <td>0.18</td>
      <td>0.12</td>
      <td>0.35</td>
      <td>0.06</td>
      <td>0.10</td>
    </tr>
    <tr style="border-bottom: 3px;">
      <td colspan="10"></td>
    </tr>
    <tr>
      <td rowspan="5" style="text-align:center; width:1%; white-space:nowrap;">
        <div style="transform:rotate(-90deg); display:inline-block;">scannet-crop <br> chair/table <br> (0.917)</div>
      </td>
      <td>Grad-CAM</td>
      <td>0.19</td>
      <td>0.31</td>
      <td>0.04</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.37</td>
      <td>0.01</td>
      <td>0.10</td>
    </tr>
    <tr>
      <td>Grad-CAM++</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.38</td>
      <td>0.01</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>HiResCAM</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>0.04</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.28</td>
      <td>0.01</td>
      <td>0.06</td>
    </tr>
    <tr>
      <td>Respond-CAM</td>
      <td>0.21</td>
      <td>0.27</td>
      <td>0.03</td>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.35</td>
      <td>0.01</td>
      <td>0.08</td>
    </tr>
    <tr>
      <td>Saliency Tubes</td>
      <td>0.52</td>
      <td>0.50</td>
      <td>0.06</td>
      <td>0.12</td>
      <td>0.07</td>
      <td>0.29</td>
      <td>0.03</td>
      <td>0.27</td>
    </tr>
    <tr style="border-bottom: 3px;">
      <td colspan="10"></td>
    </tr>
    <tr>
      <td rowspan="5" style="text-align:center; width:1%; white-space:nowrap;">
        <div style="transform:rotate(-90deg); display:inline-block;">brats-halves <br> tumor/no tumor <br> (0.796)</div>
      </td>
      <td>Grad-CAM</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>0.13</td>
      <td>0.11</td>
      <td>0.18</td>
      <td>0.00</td>
      <td>0.03</td>
    </tr>
    <tr>
      <td>Grad-CAM++</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.19</td>
      <td>0.15</td>
      <td>0.27</td>
      <td>0.03</td>
      <td>0.03</td>
    </tr>
    <tr>
      <td>HiResCAM</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.18</td>
      <td>0.19</td>
      <td>0.27</td>
      <td>0.01</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td>Respond-CAM</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.10</td>
      <td>0.07</td>
      <td>0.16</td>
      <td>0.01</td>
      <td>0.01</td>
    </tr>
    <tr>
      <td>Saliency Tubes</td>
      <td>0.19</td>
      <td>0.21</td>
      <td>0.12</td>
      <td>0.21</td>
      <td>0.14</td>
      <td>0.40</td>
      <td>0.12</td>
      <td>0.13</td>
    </tr>
  </tbody>
</table>

<table style="text-align:center;">
  <caption>Results for the <code>Mass Concentration</code> Sanity Check.</caption>
  <thead>
    <tr>
      <th></th>
      <th>Saliency Method</th>
      <th>Mass Concentration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6" style="text-align:center;">shapenet-pairs <br> Classes: airplane/bench <br> Test accuracy: 0.967</td>
      <td>Grad-CAM</td>
      <td>0.752</td>
    </tr>
    <tr>
      <td>Grad-CAM++</td>
      <td>0.727</td>
    </tr>
    <tr>
      <td>HiResCAM</td>
      <td>0.713</td>
    </tr>
    <tr>
      <td>Respond-CAM</td>
      <td>0.793</td>
    </tr>
    <tr>
      <td>SaliencyTubes</td>
      <td>0.744</td>
    </tr>
  </tbody>
</table>

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
SE3D                        # root directory
├── binvox                  # module containing the binvox tool
├── data_preprocessors      # directory containing data preprocessing classes
├── models                  # directory containing the 3D CNN model weights
├── scripts                 # directory containing the scripts for running the experiments
├── tf_saliency_methods     # module implementing the saliency methods for Keras
├── torch_saliency_methods  # module implementing the saliency methods for PyTorch
├── utils                   # directory containing utility functions
├── wsol_3d_metrics         # module implementing the evaluation metrics
├── dataset.py              # PyTorch dataset class for the customized datasets
├── LICENSE                 # license file
├── metadata.csv            # metadata file for the customized BraTS 2020 dataset
├── model.py                # 3D CNN model architecture definition
├── README.md               # this file
├── requirements.txt        # requirements file
├── run_all.sh              # script for running all experiments
└── run_benchmarks.py       # script for running the benchmarks
```

### Running the Experiments

To reproduce the experiments, execute the following shell script:

```sh
./run_all.sh --metadata_root <path_to_metadata_root> \
                --train_data_root <path_to_BraTS2020_TrainingData> \
                --datasets_root <folder_containing_customized_datasets> \
                --model_root <output_path_for_model_weights> \
```

The script will perform the following steps:

1. Create the customized BraTS 2020 dataset.
2. Train the 3D CNN model on the customized BraTS 2020 dataset on all 5 folds.
3. Evaluate the saliency methods on the trained model on all 5 folds.
4. Convert the results to *Markdown* format.

To manually reproduce the experiments, see the `run_all.sh` script for the commands to run.

You can also run the benchmarks separately using the `run_benchmarks.py` script:

```sh
python run_benchmarks.py -d <dataset> \
                          -o <results_output_dir> \
                          --datasets_base_path <path_to_datasets> \
                          --models_base_path <path_to_models> \
                          --n_folds <number_of_folds> \
                          --fold <fold_number> \
                          --train \ # boolean flag
                          --model_dataset <dataset_the_model_was_trained_on>
```

This will run the benchmarks for the saliency methods on the trained model for a chosen dataset and number of folds.

## Citation

If you find this work useful, please consider citing the following paper:

```bibtex
@INPROCEEDINGS{10647305,
  author={Wiśniewski, Mariusz and Giulivi, Loris and Boracchi, Giacomo},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={SE3D: A Framework for Saliency Method Evaluation in 3D Imaging}, 
  year={2024},
  volume={},
  number={},
  pages={89-95},
  keywords={Measurement;Deep learning;Solid modeling;Three-dimensional displays;Laser radar;Magnetic resonance imaging;Image processing;Deep Learning;Saliency Maps;3D Convolutions;Computer Vision},
  doi={10.1109/ICIP51287.2024.10647305}}
```

## Acknowledgements

This code is based on [wsolevaluation](https://github.com/clovaai/wsolevaluation), [binvox](https://github.com/faridyagubbayli/binvox), and [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam/) works.

## Code License

MIT License. See the `LICENSE` file for details.
