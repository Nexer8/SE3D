import torch
import torchio as tio
from torch.utils.data import Dataset


# Utilities for dataset folding
def fold_data_list(data_list, n_folds, test_fold, split="train"):
    fold_size = len(data_list) // n_folds
    if split == "train":
        return (
                data_list[: test_fold * fold_size]
                + data_list[(test_fold + 1) * fold_size:]
        )
    else:
        return data_list[test_fold * fold_size:(test_fold + 1) * fold_size]


class CustomDataset(Dataset):
    def __init__(self, data_list, n_folds, test_fold):
        self.data = fold_data_list(
            data_list, n_folds, test_fold, split="train")
        spatial_transforms = {
            tio.RandomElasticDeformation(): 0.2,
            tio.RandomAffine(): 0.8,
        }
        self.transform = tio.Compose([
            tio.OneOf(spatial_transforms, p=0.5),
            tio.RandomFlip(axes="lr", flip_probability=0.5)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx]["image"], dtype=torch.float)
        image = self.transform(image)
        label = torch.tensor(self.data[idx]["label"], dtype=torch.int64)
        return image, label
