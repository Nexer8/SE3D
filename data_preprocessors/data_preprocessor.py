from copy import deepcopy

import numpy as np


class DataPreprocessor:
    # For GPUs with lower memory, lower this so that large samples are excluded
    SIZE_THRESH = (0.85) * 1e8

    def __init__(self, base_path: str, low_thresh_excluded: float = 0.0, high_thresh_included: float = 0.0, mean=None,
                 std=None):
        self.base_path = base_path
        self.low_thresh_excluded = low_thresh_excluded
        self.high_thresh_included = high_thresh_included
        self.mean = mean
        self.std = std
        self.data_dict = None
        self.n_channels = None

    def load_samples(self):
        raise NotImplementedError

    def label_data(self):
        raise NotImplementedError

    def normalize_data(self):
        self.n_channels = self.data_dict[0]["image"].shape[-1]
        for elem in self.data_dict:
            elem["image"] = ((elem["image"].astype(np.double) / 255.0) - self.mean.reshape(
                [1, 1, 1, self.n_channels])) / np.sqrt(self.std).reshape([1, 1, 1, self.n_channels])
            elem["label"] = elem["label"].astype(int)

    def filter_data(self):
        filtered_datalist = []
        percents_left = []
        percents_right = []
        for elem in self.data_dict:
            value = np.sum(elem["segmentation"]) / elem["segmentation"].size
            if value <= self.low_thresh_excluded or value > self.high_thresh_included:
                filtered_datalist.append(deepcopy(elem))
                if elem["side"] == "left":
                    percents_left.append(value)
                else:
                    percents_right.append(value)
        percents = percents_left + percents_right
        # print(f"Percentages of all samples: {percents}")
        # Filter dataset based on voxel size (large samples don't fit in memory)
        prev_len = len(filtered_datalist)
        filtered_datalist = [e for e in filtered_datalist if e["image"].size <= DataPreprocessor.SIZE_THRESH]
        print(
            f"Filtered elements with size larger than {DataPreprocessor.SIZE_THRESH}, removed "
            f"{prev_len - len(filtered_datalist)} samples. Remaining {len(filtered_datalist)}")
        self.data_dict = filtered_datalist

    def transpose_data(self):
        # Transpose data to channel first for use with torch
        for elem in self.data_dict:
            if elem["image"].shape[0] == np.min(elem["image"].shape):
                print("Elements already transposed, skipping")
                break
            elem["image"] = np.transpose(elem["image"], [3, 0, 1, 2])
            if len(elem["segmentation"].shape) < 4:
                elem["segmentation"] = elem["segmentation"][np.newaxis, :, :, :]
            else:
                elem["segmentation"] = np.transpose(elem["segmentation"], [3, 0, 1, 2])

    def add_channel_dim(self):
        for sample in self.data_dict:
            if len(sample["image"].shape) == 4:
                break
            sample["image"] = sample["image"][:, :, :, np.newaxis]
            sample["segmentation"] = sample["segmentation"][:, :, :, np.newaxis]

    def preprocess(self):
        print("Loading data (1/6) - Loading", end="")
        self.load_samples()
        print(f"Loaded {len(self.data_dict)} samples")

        print("Loading data (2/6) - Adding channel dimension")
        self.add_channel_dim()

        print("Loading data (3/6) - Filtering")
        self.filter_data()

        print("Loading data (4/6) - Labeling")
        self.label_data()

        print("Loading data (5/6) - Normalizing")
        self.normalize_data()

        print("Loading data (6/6) - Transposing")
        self.transpose_data()

        return self.data_dict
