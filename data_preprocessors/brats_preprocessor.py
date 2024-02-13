from glob import glob

import numpy as np

from data_preprocessors.data_preprocessor import DataPreprocessor


class BraTSPreprocessor(DataPreprocessor):

    def __init__(self, base_path: str, mean=np.array([0.04844053, 0.06671674, 0.04088918, 0.04963282]),
                 std=np.array([0.11645673, 0.15704592, 0.09690811, 0.12228158]), high_thresh_included=0.003):
        super().__init__(base_path=base_path, mean=mean, std=std, high_thresh_included=high_thresh_included)

    def load_samples(self):
        # All samples  (240, 120, 155, 4)
        dataset_path = f"{self.base_path}/BraTS"

        def create_sample(base_path, side):
            sample_stem = base_path.split("/")[-1]
            return {
                "image": np.load(f"{base_path}/{side}/{sample_stem}_{side}.npz")["data"],
                "segmentation": np.load(f"{base_path}/{side}/{sample_stem}_seg_{side}.npz")["data"],
                "side": side
            }

        self.data_dict = [create_sample(sample, side) for sample in glob(f"{dataset_path}/*") for side in
                          ["left", "right"]]

    def label_data(self):
        for sample in self.data_dict:
            value = np.sum(sample["segmentation"]) / sample["segmentation"].size
            sample["label"] = value > 0
