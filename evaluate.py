import argparse
from glob import glob
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from tf_saliency_methods import GradCAM, GradCAMPlusPlus, HiResCAM, SaliencyTubes
from tf_saliency_methods.utils import find_last_layer_by_type
from wsol_3d_metrics import BBoxEvaluator, BBoxEvaluator3D, MaskEvaluator


def evaluate(base_path: str, model_path: str, fold: int, iou_threshold_list: Tuple[float]):
    df = pd.read_csv(f'{base_path}/metadata.csv')

    # Parameters
    last_conv_layer_type = tf.keras.layers.Conv3D
    classes = {'Normal': 0, 'Tumor': 1}

    # Building the Test Set
    labels = np.array(df['label'])
    paths = np.array(df['path'])
    train_indices = df[df['fold'] != fold].index.values.tolist()
    test_indices = df[df['fold'] == fold].index.values.tolist()
    assert set(train_indices).isdisjoint(test_indices)

    # Filter out the Normal cases
    test_indices = [i for i in test_indices if labels[i] == 1]

    # Evaluation Loop
    for method in [GradCAM, HiResCAM, GradCAMPlusPlus, SaliencyTubes]:
        # Load the model
        model = tf.keras.models.load_model(
            f'{model_path}/fold{fold}/best{fold}.h5',
        )

        # Find the last conv layer
        last_conv_layer_name = find_last_layer_by_type(last_conv_layer_type, model)
        # Initialize the CAM method
        cam_method = method(model, last_conv_layer_name=last_conv_layer_name)

        # Initialize the evaluators

        max_box_acc_evaluator = BBoxEvaluator(
            iou_threshold_list=iou_threshold_list,
            multi_contour_eval=False,
        )
        max_box_acc_v2_evaluator = BBoxEvaluator(
            iou_threshold_list=iou_threshold_list,
            multi_contour_eval=True,
        )
        vx_ap_evaluator = MaskEvaluator(
            iou_threshold_list=iou_threshold_list,
        )
        max_3d_box_acc_evaluator = BBoxEvaluator3D(
            iou_threshold_list=iou_threshold_list,
            multi_contour_eval=False,
        )
        max_3d_box_acc_v2_evaluator = BBoxEvaluator3D(
            iou_threshold_list=iou_threshold_list,
            multi_contour_eval=True,
        )

        # Iterate over the test set
        for idx in test_indices:
            vol_paths = list(glob(f'{base_path}/{paths[idx]}/*.npz'))

            vol_path = [path for path in vol_paths if 'seg' not in path][0]
            volume = np.load(vol_path)['data']
            volume = (volume.astype('float32') / 255.0).astype('float32')

            mask_path = [path for path in vol_paths if 'seg' in path][0]
            mask = np.load(mask_path)['data'].astype('float32')

            # compute the heatmap
            heatmap = cam_method.get_cam(np.expand_dims(volume, axis=0), classes['Tumor'])
            # accumulate the metrics
            vx_ap_evaluator.accumulate(heatmap, mask)
            max_3d_box_acc_evaluator.accumulate(heatmap, mask)
            max_3d_box_acc_v2_evaluator.accumulate(heatmap, mask)

            # iterate over slices of the 3D heatmap
            for i in range(heatmap.shape[-1]):
                max_box_acc_evaluator.accumulate(heatmap[..., i], mask[..., i])
                max_box_acc_v2_evaluator.accumulate(heatmap[..., i], mask[..., i])

        # compute the metrics
        performances = {
            'MaxBoxAcc': max_box_acc_evaluator.compute(),
            'MaxBoxAccV2': max_box_acc_v2_evaluator.compute(),
            'VxAP': vx_ap_evaluator.compute(),
            'Max3DBoxAcc': max_3d_box_acc_evaluator.compute(),
            'Max3DBoxAccV2': max_3d_box_acc_v2_evaluator.compute()
        }

        # save the metrics
        np.savez_compressed(f'{model_path}/fold{fold}/{method.__name__}.npz', data=performances)
        print(f'{method.__name__} done!')
        print(performances)
        print('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_root', type=str, default='.',
                        help="Root folder of metadata.")
    parser.add_argument('--model_path', type=str, default='.',
                        help="Root folder of the model to evaluate.")
    parser.add_argument('--fold', type=int, default=0,
                        help="Fold to evaluate.")
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=(30, 50, 70))
    args = parser.parse_args()

    evaluate(
        base_path=args.metadata_root,
        model_path=args.model_path,
        fold=args.fold,
        iou_threshold_list=args.iou_threshold_list
    )


if __name__ == "__main__":
    main()
