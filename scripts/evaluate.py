import numpy as np
import torch
from tqdm import tqdm

from torch_saliency_methods import SaliencyTubes
from torch_saliency_methods.utils.model_targets import ClassifierOutputTarget
from utils.data_utils import fold_data_list
from wsol_3d_metrics import (BBoxEvaluator, BBoxEvaluator3D, F1Evaluator,
                             F1OnlyMaskEvaluator, MaskEvaluator,
                             MassConcentrationEvaluator)


def compute_metrics(bench_config, eval_config, model, pred_layer, target_layer, data_dict):
    print("Computing WSOL and WSSS metrics (1/1)")
    with open(bench_config['output_path'], "w") as write_file:
        for method in eval_config['CAM_METHODS']:
            print()
            print(method)
            write_file.write(f"\nCAM: {method}\n")
            # From 0 to 1 both exclusive
            # thresholds = list(np.linspace(start=0, stop=1, num=N_THRESH_TAU + 1, endpoint=False))[1:]
            if method == SaliencyTubes:
                cam = method(model=model, target_layers=[target_layer], pred_layer=pred_layer)
            else:
                cam = method(model=model, target_layers=[target_layer])

            MaxBoxAcc_evaluator = BBoxEvaluator(
                iou_threshold_list=eval_config['DELTA_ACC_THRESH_V1'],
                multi_contour_eval=False,
            )
            MaxBoxAccV2_evaluator = BBoxEvaluator(
                iou_threshold_list=eval_config['DELTA_ACC_THRESH_V2'],
                multi_contour_eval=True,
            )
            # It's the same also as PxAP
            VxAP_evaluator = MaskEvaluator(
                iou_threshold_list=eval_config['DELTA_ACC_THRESH_V1'],
            )
            Max3DBoxAcc_evaluator = BBoxEvaluator3D(
                iou_threshold_list=eval_config['DELTA_ACC_THRESH_V1'],
                multi_contour_eval=False,
            )
            Max3DBoxAccV2_evaluator = BBoxEvaluator3D(
                iou_threshold_list=eval_config['DELTA_ACC_THRESH_V2'],
                multi_contour_eval=True,
            )
            F1_Evaluator = F1Evaluator()
            F1MaskOnly_Evaluator = F1OnlyMaskEvaluator()
            MassConcentration_Evaluator = MassConcentrationEvaluator()

            evaluators = [
                {
                    "name": "MaxBoxAcc",
                    "evaluator_f": MaxBoxAcc_evaluator,
                    "3D": False,
                },
                {
                    "name": "MaxBoxAccV2",
                    "evaluator_f": MaxBoxAccV2_evaluator,
                    "3D": False,
                },
                {
                    "name": "VxAP",
                    "evaluator_f": VxAP_evaluator,
                    "3D": True,
                },
                {
                    "name": "Max3DBoxAcc",
                    "evaluator_f": Max3DBoxAcc_evaluator,
                    "3D": True,
                },
                {
                    "name": "Max3DBoxAccV2",
                    "evaluator_f": Max3DBoxAccV2_evaluator,
                    "3D": True,
                },
                {
                    "name": "F1 (F1Score, tau_f1, (Prec@tau_f1, Rec@tau_f1))",
                    "evaluator_f": F1_Evaluator,
                    "3D": True,
                },
            ]

            # Mass Concentration is only for shapenet-pairs
            if bench_config['dataset'] == "shapenet-pairs":
                evaluators.append({
                    "name": "MassConcentration",
                    "evaluator_f": MassConcentration_Evaluator,
                    "3D": True,
                })

            # Compute cams for all samples
            all_cams = []
            test_samples = fold_data_list(data_dict, n_folds=bench_config['n_folds'], test_fold=bench_config['fold'],
                                          split="test")
            n_samples = eval_config['N_SAMPLES']
            if eval_config['N_SAMPLES'] < 1:
                n_samples = len(test_samples)
                print(f"Executing on all {n_samples} test samples")
            if eval_config['N_SAMPLES'] > len(test_samples):
                raise ValueError(
                    f"N_SAMPLES={eval_config['N_SAMPLES']} is greater than the available number of samples={len(test_samples)}")
            test_samples = test_samples[:n_samples]
            for sample in tqdm(test_samples):
                image_in = torch.tensor(sample["image"], dtype=torch.float).cuda()
                target_label = int(sample["label"])

                targets = [ClassifierOutputTarget(target_label)]
                grayscale_cam = cam(input_tensor=image_in, targets=targets).astype(np.float32)
                if np.sum(grayscale_cam == 0):
                    grayscale_cam[0, 0, 0] = np.finfo(float).eps
                normalized_cam = (grayscale_cam - np.min(grayscale_cam)) / (
                        np.max(grayscale_cam) - np.min(grayscale_cam))
                all_cams.append(normalized_cam)

            # Compute metrics for computed CAMs
            for evaluator in evaluators:
                print(f"Computing {evaluator['name']}")
                # If the evaluator function supports 3D, then call its accumulator function, else call it on each slice
                # NOTE: Due to accumulation over slices, the result may not be a multiple of 1/N_SAMPLES

                if evaluator["3D"]:
                    def accumulate(heatmap, mask):
                        evaluator["evaluator_f"].accumulate(heatmap, mask)
                else:
                    def accumulate(heatmap, mask):
                        for i in range(heatmap.shape[-1]):
                            evaluator["evaluator_f"].accumulate(heatmap[..., i], mask[..., i])

                for idx, normalized_cam in tqdm(enumerate(all_cams)):
                    accumulate(
                        normalized_cam, test_samples[idx]["segmentation"][0].astype(np.float32))

                print(
                    f"{evaluator['name']}: {evaluator['evaluator_f'].compute()}")
                write_file.write(f"{evaluator['name']}: {evaluator['evaluator_f'].compute()}\n")
