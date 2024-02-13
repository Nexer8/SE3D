import os
import random
from glob import glob

import numpy as np

from binvox import binvox
from data_preprocessors.data_preprocessor import DataPreprocessor


def load_binvox(path):
    return (
            binvox.Binvox.read(path, mode='dense').numpy().astype(np.uint8) * 255
    )


def load_all_from_folder_list(folder_list):
    return [(load_binvox(file), file.split("/")[-3]) for folder in folder_list for file in
            glob(f"{str(folder)}/**/*.binvox", recursive=True)]


def load_class_dict(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    return {line.split(" ")[0]: line.split(" ")[1] for line in lines}


class ShapenetBasePreprocessor(DataPreprocessor):
    def __init__(self, base_path: str, mean=np.array([0.06027088]), std=np.array([0.23798802]),
                 train_class_ids: tuple = None):
        super().__init__(base_path=base_path, mean=mean, std=std)
        assert len(train_class_ids) == 2
        self.train_class_ids = train_class_ids


class ShapenetBinaryPreprocessor(ShapenetBasePreprocessor):
    # chair / table
    def __init__(self, base_path: str, train_class_ids: tuple = ("03001627", "04379243")):
        super().__init__(base_path=base_path, train_class_ids=train_class_ids)

    def load_samples(self):
        # All samples shapenet-binary (32, 32, 32, 1)
        dataset_path = f"{self.base_path}/ShapeNetVox32"

        class_folders = os.listdir(dataset_path)
        class_folders = [e for e in class_folders if os.path.isdir(f"{dataset_path}/{e}")]

        train_class_folders = [f"{dataset_path}/{e}" for e in class_folders if e in self.train_class_ids]
        train_data = load_all_from_folder_list(train_class_folders)

        shapenet_data = [
            {
                "image": train_sample,
                "segmentation": train_sample // 255,
                "label": np.array(self.train_class_ids.index(label), dtype=np.int64),
                "side": None,
            }
            for train_sample, label in train_data
        ]
        random.shuffle(shapenet_data)
        self.data_dict = shapenet_data

    def label_data(self):
        pass


class ShapenetPairsPreprocessor(ShapenetBasePreprocessor):
    # airplane / bench
    def __init__(self, base_path: str, train_class_ids: tuple = ("02691156", "02828884")):
        super().__init__(base_path=base_path, train_class_ids=train_class_ids)

    def load_samples(self):
        # All samples shapenet-pairs (64, 32, 32, 1)
        dataset_path = f"{self.base_path}/ShapeNetVox32"

        class_folders = os.listdir(dataset_path)
        class_folders = [e for e in class_folders if os.path.isdir(f"{dataset_path}/{e}")]

        train_class_folders = [f"{dataset_path}/{e}" for e in class_folders if e in self.train_class_ids]
        noise_class_folders = [f"{dataset_path}/{e}" for e in class_folders if e not in self.train_class_ids]

        train_data = load_all_from_folder_list(train_class_folders)
        noise_data = load_all_from_folder_list(noise_class_folders)

        shapenet_data = []
        for train_sample, label in train_data:
            noise_id = np.random.randint(0, len(noise_data))
            noise_sample = noise_data[noise_id][0]
            left_right = np.random.randint(0, 2)
            if left_right == 0:
                cat_sample = np.concatenate((train_sample, noise_sample), axis=0)
                cat_seg_mask = np.concatenate((train_sample // 255, np.zeros_like(noise_sample)), axis=0)
                side = "left"
            else:
                cat_sample = np.concatenate((noise_sample, train_sample), axis=0)
                cat_seg_mask = np.concatenate((np.zeros_like(noise_sample), train_sample // 255), axis=0)
                side = "right"

            dict_sample = {"image": cat_sample, "segmentation": cat_seg_mask, "label": np.array(
                self.train_class_ids.index(label), dtype=np.int64), "side": side}
            shapenet_data.append(dict_sample)

        random.shuffle(shapenet_data)
        self.data_dict = shapenet_data

    def label_data(self):
        pass
