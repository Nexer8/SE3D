import numpy as np
from tqdm import tqdm


def fold_data_list(data_list, n_folds, test_fold, split="train"):
    fold_size = len(data_list) // n_folds
    if split == "train":
        return (
                data_list[: test_fold * fold_size]
                + data_list[(test_fold + 1) * fold_size:]
        )
    else:
        return data_list[test_fold * fold_size:(test_fold + 1) * fold_size]


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
