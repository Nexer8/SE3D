# WSOL 3D Metrics

Author: `REDACTED` *(available upon publication)*

This repository contains the code for the evaluation metrics introduced in the paper *"SE3D: A Framework for Saliency
Method Evaluation in 3D Medical Imaging"*.

## Table of Contents

- [Available Metrics](#available-metrics)
- [Usage](#usage)
    - [Requirements](#requirements)
    - [Code Structure](#code-structure)
    - [Example](#example)
- [Code License](#code-license)

## Available Metrics

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

## Usage

To use the metrics in your project, you can install this repository as a pip package:

```sh
pip install -e wsol_3d_metrics
```

### Requirements

The requirements for this repository are listed in `requirements.txt`. In case you want to install the package manually,
you can do so by running:

```sh
pip install -r requirements.txt
```

### Code Structure

The code is structured as follows:

```sh
wsol_3d_metrics                 # root directory
├── BBoxEvaluator.py            # implements MaxBoxAcc and MaxBoxAccV2
├── BBoxEvaluator3D.py          # implements Max3DBoxAcc and Max3DBoxAccV2
├── LocalizationEvaluator.py    # base class for all metrics
├── MaskEvaluator.py            # implements VxAP
├── README.md                   # this file
├── __init__.py                 # imports all metrics
├── requirements.txt            # requirements for this repository
└── setup.py                    # setup file for pip
```

### Example

```py
import numpy as np
from wsol_3d_metrics import BBoxEvaluator3D, MaskEvaluator

# ground truth, 3x3x3 box in the top left corner
# and 4x4x2 box in the bottom right corner
gt = np.zeros((20, 20, 20), dtype=np.float32)
gt[:3, :3, :3] = 1
gt[-4:, -4:, -2:] = 1

# evaluator for Max3DBoxAcc
max_3d_box_acc_evaluator = BBoxEvaluator3D(
    iou_threshold_list=(30, 50, 70),
    multi_contour_eval=False,
)

# evaluator for Max3DBoxAccV2
max_3d_box_acc_v2_evaluator = BBoxEvaluator3D(
    iou_threshold_list=(30, 50, 70),
    multi_contour_eval=True,
)

# evaluator for VxAP
vxap_evaluator = MaskEvaluator()

# prediction, 3x3x3 box in the top left corner
pred = np.zeros((20, 20, 20), dtype=np.float32)
pred[:3, :3, :3] = 0.9

max_3d_box_acc_evaluator.accumulate(pred, gt)
max_3d_box_acc_v2_evaluator.accumulate(pred, gt)
vxap_evaluator.accumulate(pred, gt)

print(f'Max3DBoxAcc:', max_3d_box_acc_evaluator.compute())
print(f'Max3DBoxAccV2:', max_3d_box_acc_v2_evaluator.compute())
print(f'VxAP:', vxap_evaluator.compute())
```

To see more examples, please refer to the `examples/` directory.

## Code License

To be determined.
