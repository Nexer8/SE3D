import os
import random
from glob import glob
from pathlib import Path

import binvox.binvox as binvox
import numpy as np
import open3d as o3d
from tqdm import tqdm

from pcd_utils import load_ply_scene, filter_by_object, crop_point_cloud, voxelize_point_cloud


def compute_data_stats(data_list):
    # check that it's channel last
    assert data_list[0]["image"].shape[-1] == np.min(data_list[0]["image"].shape)
    # compute n channels
    image_data = [e["image"] for e in data_list]
    n_channels = data_list[0]["image"].shape[-1]
    sums = np.zeros([n_channels])
    elem_count = 0
    for elem in tqdm(image_data):
        sums += np.sum(elem, axis=(0, 1, 2))
        elem_count += np.size(elem) // n_channels

    means = sums / elem_count

    deviations = np.zeros([n_channels])
    for elem in tqdm(image_data):
        deviations += np.sum((elem - means) ** 2, axis=(0, 1, 2))

    deviations = np.sqrt(deviations / elem_count)
    return means.flatten() / 255.0, deviations.flatten() / 255.0


def load_BraTS_samples(base_folder):
    def create_sample(base_path, side):
        sample_stem = base_path.split("/")[-1]
        return {
            "image": np.load(Path(base_path) / f"{side}/{sample_stem}_{side}.npz")["data"],
            "segmentation": np.load(Path(base_path) / f"{side}/{sample_stem}_seg_{side}.npz")["data"],
            "side": side
        }

    samples_folders = glob(f"{str(base_folder)}/*")
    # Each folder has a left and right sample
    return [create_sample(sample, side) for sample in samples_folders for side in ["left", "right"]]


def load_KiTS_samples(base_folder):
    def create_sample(filepath, side):
        seg_id = filepath.split("/")[-1].split("-")[-1]
        folder_path = "/".join(filepath.split("/")[:-1])
        return {
            "image": np.load(filepath)["data"],
            "segmentation": np.load(Path(folder_path) / f"segmentation-{seg_id}")["data"],
            "side": side
        }

    samples_left = glob(f"{str(base_folder)}/left/volume*")
    samples_right = glob(f"{str(base_folder)}/right/volume*")

    return ([create_sample(sample, "left") for sample in samples_left]
            + [create_sample(sample, "right") for sample in samples_right])


def load_lung_samples(base_folder):
    return load_KiTS_samples(base_folder)


def load_binvox(path):
    return (
            binvox.Binvox.read(path, mode='dense').numpy().astype(np.uint8) * 255
    )


def load_all_from_folder_list(folder_list):
    return [(load_binvox(file), file.split("/")[-3]) for folder in folder_list for file in
            glob(f"{str(folder)}/**/*.binvox", recursive=True)]


def load_shapenet_pair_samples(base_folder, train_classes_ids):
    # creates pairs where one sample is from train classes and one sample is from other classes
    assert len(train_classes_ids) == 2

    class_folders = os.listdir(base_folder)
    class_folders = [e for e in class_folders if os.path.isdir(f"{base_folder}/{e}")]

    train_class_folders = [f"{base_folder}/{e}" for e in class_folders if e in train_classes_ids]
    noise_class_folders = [f"{base_folder}/{e}" for e in class_folders if e not in train_classes_ids]

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
            train_classes_ids.index(label), dtype=np.int64), "side": side}
        shapenet_data.append(dict_sample)

    random.shuffle(shapenet_data)
    return shapenet_data


def load_shapenet_binary(base_folder, train_classes_ids):
    # creates a subset of shapenet for classification over the train_classes_ids
    assert len(train_classes_ids) == 2

    class_folders = os.listdir(base_folder)
    class_folders = [e for e in class_folders if os.path.isdir(f"{base_folder}/{e}")]

    train_class_folders = [f"{base_folder}/{e}" for e in class_folders if e in train_classes_ids]
    train_data = load_all_from_folder_list(train_class_folders)

    shapenet_data = [
        {
            "image": train_sample,
            "segmentation": train_sample // 255,
            "label": np.array(train_classes_ids.index(label), dtype=np.int64),
            "side": None,
        }
        for train_sample, label in train_data
    ]
    random.shuffle(shapenet_data)
    return shapenet_data


def shapenet_class_dict(assignment_file_path):
    with open(assignment_file_path) as f:
        lines = f.readlines()

    return {line.split(" ")[0]: line.split(" ")[1] for line in lines}


def load_scannet(base_folder, train_classnames, isolated):
    # Get all downloaded scene names
    # All scenes might have more than one scan, but we want unique scenes, so only take the one with "_00"
    scan_paths = glob(f"{str(base_folder)}/scans/scene*/*_vh_clean_2.labels.ply")
    scan_names = [f"{e}_00" for e in list({"_".join(e.split("/")[-1].split("_")[:1]) for e in scan_paths})]
    scannet_data = []

    # Iterate over all scans
    for scan_name in tqdm(scan_names):
        pt_cloud, class_ids, obj_classes, vert_classes, vert_objects = load_ply_scene(base_folder, scan_name)
        # Iterate over train classes
        for label, classname in enumerate(train_classnames):
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
                if not isolated:
                    # Find bounds of object and crop point cloud
                    vertices = np.asarray(filtered_point_cloud.points)
                    min_bound = np.min(vertices, axis=0)
                    max_bound = np.max(vertices, axis=0)
                    mask_point_cloud = o3d.geometry.PointCloud(filtered_point_cloud)
                    filtered_point_cloud = crop_point_cloud(pt_cloud, min_bound, max_bound)
                else:
                    mask_point_cloud = filtered_point_cloud

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
    return scannet_data


def load_scannet_isolated(base_folder, train_classnames):
    return load_scannet(base_folder, train_classnames, isolated=True)


def load_scannet_crop(base_folder, train_classnames):
    return load_scannet(base_folder, train_classnames, isolated=False)
