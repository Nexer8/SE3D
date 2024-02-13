from glob import glob

import numpy as np

from data_preprocessors.data_preprocessor import DataPreprocessor


class MDLungPreprocessor(DataPreprocessor):
    def __init__(self, base_path: str, mean=np.array([0.11019484]), std=np.array([0.12821062]),
                 high_thresh_included=0.003):
        super().__init__(base_path=base_path, mean=mean, std=std, high_thresh_included=high_thresh_included)

    def load_samples(self):
        # Samples between (512, 256, 112, 1) and (512, 256, 636, 1)
        dataset_path = f"{self.base_path}/decathlon_lungs"

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
