import random
from glob import glob

import numpy as np
import open3d as o3d
from tqdm import tqdm

from data_preprocessors.data_preprocessor import DataPreprocessor
from utils.pcd_utils import load_ply_scene, filter_by_object, voxelize_point_cloud, crop_point_cloud


class ScannetBasePreprocessor(DataPreprocessor):
    def __init__(self, base_path: str, mean=np.array([0.02899467]), std=np.array([0.16779148]),
                 train_class_names: tuple = ("chair", "table")):
        super().__init__(base_path=base_path, mean=mean, std=std)
        assert len(train_class_names) == 2
        self.train_class_names = train_class_names


class ScannetIsolatedPreprocessor(ScannetBasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_samples(self):
        # All samples (32, 32, 32, 1)
        dataset_path = f"{self.base_path}/scannet"

        # Get all downloaded scene names
        # All scenes might have more than one scan, but we want unique scenes, so only take the one with "_00"
        scan_paths = glob(f"{dataset_path}/scans/scene*/*_vh_clean_2.labels.ply")
        scan_names = [f"{e}_00" for e in list({"_".join(e.split("/")[-1].split("_")[:1]) for e in scan_paths})]
        scannet_data = []

        # Iterate over all scans
        for scan_name in tqdm(scan_names):
            pt_cloud, class_ids, obj_classes, vert_classes, vert_objects = load_ply_scene(dataset_path, scan_name)
            # Iterate over train classes
            for label, classname in enumerate(self.train_class_names):
                # Find all object ids in pcd that have class = classname
                try:
                    search_class_objects = [idx for idx, class_id in enumerate(obj_classes) if
                                            class_ids[classname] == class_id]
                except KeyError:
                    print(f"Scan {scan_name} has no {classname} objects, skipping.")
                    continue

                # Iterate over object of type classname
                for search_class_object in search_class_objects:
                    # Filter point cloud based on object and voxelize it
                    filtered_point_cloud = filter_by_object(pt_cloud, vert_objects, search_class_object)
                    if len(filtered_point_cloud.points) == 0:
                        # If the object is empty
                        continue

                    voxelized_mask = voxelize_point_cloud(filtered_point_cloud)

                    # rotate to y-up for consistency w/ shapenet
                    voxelized_mask = np.transpose(voxelized_mask, axes=(0, 2, 1))

                    dict_sample = {
                        "image": voxelized_mask * 255,
                        "segmentation": voxelized_mask,
                        "label": np.array(label, dtype=np.int64),
                        "side": None,
                    }
                    scannet_data.append(dict_sample)

        random.shuffle(scannet_data)
        self.data_dict = scannet_data

    def label_data(self):
        pass


class ScannetCropPreprocessor(ScannetBasePreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_samples(self):
        # All samples (32, 32, 32, 1)
        dataset_path = f"{self.base_path}/scannet"

        # Get all downloaded scene names
        # All scenes might have more than one scan, but we want unique scenes, so only take the one with "_00"
        scan_paths = glob(f"{str(dataset_path)}/scans/scene*/*_vh_clean_2.labels.ply")
        scan_names = [f"{e}_00" for e in list({"_".join(e.split("/")[-1].split("_")[:1]) for e in scan_paths})]
        scannet_data = []

        # Iterate over all scans
        for scan_name in tqdm(scan_names):
            pt_cloud, class_ids, obj_classes, vert_classes, vert_objects = load_ply_scene(dataset_path, scan_name)
            # Iterate over train classes
            for label, classname in enumerate(self.train_class_names):
                # Find all object ids in pcd that have class = classname
                try:
                    search_class_objects = [idx for idx, class_id in enumerate(obj_classes) if
                                            class_ids[classname] == class_id]
                except KeyError:
                    print(f"Scan {scan_name} has no {classname} objects, skipping.")
                    continue

                # Iterate over object of type classname
                for search_class_object in search_class_objects:
                    # Filter point cloud based on object and voxelize it
                    filtered_point_cloud = filter_by_object(pt_cloud, vert_objects, search_class_object)
                    if len(filtered_point_cloud.points) == 0:
                        # If the object is empty
                        continue

                    # Find bounds of object and crop point cloud
                    vertices = np.asarray(filtered_point_cloud.points)
                    min_bound = np.min(vertices, axis=0)
                    max_bound = np.max(vertices, axis=0)
                    mask_point_cloud = o3d.geometry.PointCloud(filtered_point_cloud)
                    filtered_point_cloud = crop_point_cloud(pt_cloud, min_bound, max_bound)

                    voxelized_image = voxelize_point_cloud(filtered_point_cloud)
                    voxelized_mask = voxelize_point_cloud(mask_point_cloud)
                    # rotate to y-up for consistency w/ shapenet
                    voxelized_image = np.transpose(voxelized_image, axes=(0, 2, 1))
                    voxelized_mask = np.transpose(voxelized_mask, axes=(0, 2, 1))

                    dict_sample = {
                        "image": voxelized_image * 255,
                        "segmentation": voxelized_mask,
                        "label": np.array(label, dtype=np.int64),
                        "side": None,
                    }
                    scannet_data.append(dict_sample)

        random.shuffle(scannet_data)
        self.data_dict = scannet_data

    def label_data(self):
        pass
