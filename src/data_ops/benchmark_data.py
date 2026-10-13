"""Benchmark loading with the same simulated client split used for cancer.

Source splits are pooled before the framework creates its own client holdouts.
"""

import numpy as np
import random

from src.config import constants
from src.data_ops.iid_partition import partition_iid_indices
from src.data_ops.attacks import assign_fake_clients


def load_benchmark_arrays(dataset_name, max_samples, seed):
    if dataset_name == "mnist":
        from src.data_ops.mnist_data import load_mnist_arrays
        images, labels = load_mnist_arrays()
    else:
        images, labels = _load_tfds_arrays(dataset_name)
    # Match cancer: seeded global shuffle, limit, contiguous client partitions.
    indices = list(range(len(labels)))
    random.Random(seed).shuffle(indices)
    if max_samples is not None:
        indices = indices[:max_samples]
    return images[indices], labels[indices]


def _load_tfds_arrays(dataset_name):
    import tensorflow_datasets as tfds

    names = {"emnist": "emnist/balanced", "cifar10": "cifar10"}
    dataset = tfds.load(
        names[dataset_name], split="train+test", as_supervised=True,
        shuffle_files=False, data_dir=constants.DATASET_CACHE_DIR,
    )
    images, labels = zip(*tfds.as_numpy(dataset))
    images = np.stack(images).astype(np.float32) / 255.0
    if dataset_name == "emnist":
        # TFDS stores EMNIST with transposed spatial axes.
        images = images.transpose(0, 2, 1, 3)
    return images, np.asarray(labels, dtype=np.int32)


def prepare_benchmark_clients(
    dataset_name, num_client, num_malicious, attack_list, attack_params,
    seed, max_samples, attack_distribution, test_split, batch_size,
):
    from src.data_ops.client_partition import create_client_dataset

    if num_client < 1 or batch_size < 1 or not 0 < test_split < 1:
        raise ValueError("Use positive client/batch counts and 0 < TEST_SPLIT < 1.")
    if max_samples is not None and max_samples < 2 * num_client:
        raise ValueError("MAX_SAMPLES must allow at least two examples per client.")
    images, labels = load_benchmark_arrays(dataset_name, max_samples, seed)
    if len(labels) < 2 * num_client:
        raise ValueError("The dataset must contain at least two examples per client.")
    malicious, attack_map, client_ids = assign_fake_clients(
        num_client=num_client, num_malicious=num_malicious,
        attack_list=attack_list, seed=seed, exclude_head=True,
        max_fake_ratio=constants.MAX_FAKE_RATIO,
        attack_distribution=attack_distribution,
    )
    partitions = partition_iid_indices(labels, num_client, seed)
    train, test = {}, {}
    if dataset_name == "mnist":
        class_names = [str(digit) for digit in range(10)]
    elif dataset_name == "cifar10":
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
    else:
        # EMNIST balanced: digits, uppercase letters, then 11 lowercase classes.
        class_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")
    for i, client_id in enumerate(client_ids):
        indices = partitions[i]
        train[client_id], test[client_id] = create_client_dataset(
            image_paths=None, labels=labels[indices], client_id=client_id,
            malicious_clients_attack_map=attack_map, attack_params=attack_params,
            image_arrays=images[indices], seed=seed + i,
            batch_size=batch_size, test_split=test_split,
            class_names=class_names,
        )
    return train, test, malicious, attack_map
