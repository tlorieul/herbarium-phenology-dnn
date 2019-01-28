import numpy as np


def get_random_train_val_split_indices(
        train_dataset, test_size=.2, shuffle=False, random_seed=0):
    n_samples = len(train_dataset)
    split = int(test_size * n_samples)
    indices = range(n_samples)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_ind, val_ind = indices[:-split], indices[-split:]
    return train_ind, val_ind
