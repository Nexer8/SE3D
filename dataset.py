import torch
import torchio as tio
from torch.utils.data import Dataset

from utils.data_utils import fold_data_list


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
