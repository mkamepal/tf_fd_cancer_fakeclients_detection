"""Disjoint, reproducible IID allocations before splitting or poisoning."""

import numpy as np
from src.config import constants


def partition_iid_indices(labels, num_clients, seed=42):
    """Compose the IID switches in constants.py into one client allocation.

    Balance the source pool first, choose equal or unequal client capacities,
    then use random allocation or proportional stratification. Stratification
    matches class proportions up to integer rounding; finite random samples
    only approximate the pool distribution. No sample is shared by clients.
    """
    techniques = constants.get_iid_techniques()
    labels = np.asarray(labels)
    if labels.ndim != 1 or num_clients < 1:
        raise ValueError("Use one-dimensional labels and a positive client count.")
    if len(labels) < 2 * num_clients:
        raise ValueError("The dataset must contain at least two examples per client.")
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(labels, return_counts=True)
    groups = [rng.permutation(np.flatnonzero(labels == label)) for label in classes]
    if "balanced" in techniques:
        groups = [group[:counts.min()] for group in groups]
    counts = np.array([len(group) for group in groups])
    total = int(counts.sum())
    if total < 2 * num_clients:
        raise ValueError("Balanced IID leaves fewer than two examples per client.")
    if "unequal" in techniques:
        alpha = constants.IID_UNEQUAL_ALPHA
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("IID_UNEQUAL_ALPHA must be finite and positive.")
        sizes = 2 + rng.multinomial(
            total - 2 * num_clients,
            rng.dirichlet(np.full(num_clients, alpha)),
        )
    else:
        sizes = np.diff(np.arange(num_clients + 1) * total // num_clients)
    # Balanced alone implements the balanced-class baseline. Explicit RANDOM
    # with BALANCED permits random draws from the balanced pool instead.
    stratified = "stratified" in techniques or (
        "balanced" in techniques and "random" not in techniques
    )
    if not stratified:
        indices = rng.permutation(np.concatenate(groups))
        return list(np.split(indices, np.cumsum(sizes)[:-1]))

    # Round a proportional class/client table while preserving both margins.
    ideal = np.outer(counts, sizes) / total
    allocation = np.floor(ideal).astype(int)
    row_remaining = counts - allocation.sum(axis=1)
    col_remaining = sizes - allocation.sum(axis=0)
    while row_remaining.sum():
        scores = ideal - allocation
        scores[row_remaining == 0, :] = -np.inf
        scores[:, col_remaining == 0] = -np.inf
        row, col = np.unravel_index(np.argmax(scores), scores.shape)
        allocation[row, col] += 1
        row_remaining[row] -= 1
        col_remaining[col] -= 1
    buckets = [[] for _ in range(num_clients)]
    for group, class_sizes in zip(groups, allocation):
        for client, part in enumerate(np.split(group, np.cumsum(class_sizes)[:-1])):
            buckets[client].extend(part)
    return [rng.permutation(bucket).astype(np.int64) for bucket in buckets]
