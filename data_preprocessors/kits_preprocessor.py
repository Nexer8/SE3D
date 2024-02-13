from glob import glob

import numpy as np

from data_preprocessors.data_preprocessor import DataPreprocessor


class KiTSPreprocessor(DataPreprocessor):
    def __init__(self, base_path: str, mean=np.array([0.18517924]), std=np.array([0.18119358]),
                 high_thresh_included=0.001):
        super().__init__(base_path=base_path, mean=mean, std=std, high_thresh_included=high_thresh_included)

    def load_samples(self):
        # Samples between (512, 256, 29, 1) and (512, 398, 1059, 1)
        dataset_path = f"{self.base_path}/KiTS"

        def create_sample(filepath, side):
            seg_id = filepath.split("/")[-1].split("-")[-1]
            folder_path = "/".join(filepath.split("/")[:-1])
            return {
                "image": np.load(filepath)["data"],
                "segmentation": np.load(f"{folder_path}/segmentation-{seg_id}")["data"],
                "side": side
            }

        self.data_dict = [create_sample(sample, "left") for sample in glob(f"{dataset_path}/left/volume*")] + [
            create_sample(sample, "right") for sample in glob(f"{dataset_path}/right/volume*")]

    def label_data(self):
        for sample in self.data_dict:
            value = np.sum(sample["segmentation"]) / sample["segmentation"].size
            sample["label"] = value > 0
