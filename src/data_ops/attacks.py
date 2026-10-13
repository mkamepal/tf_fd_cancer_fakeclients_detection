import random
import numpy as np
import tensorflow as tf


def label_flip_attack(labels, flip_prob, seed=42):

    labels = np.array(labels,dtype=np.int32).copy()
    rng = np.random.RandomState(seed)
    n = len(labels)

    classes = np.unique(labels)
    num_classes = len(classes)
    if n == 0 or num_classes < 2:
        return labels

    mask = rng.rand(n) < float(flip_prob)

    if mask.sum() > 0:
        # Local multiclass partitions may contain non-contiguous labels.
        # Choose a different observed class without inventing class IDs.
        source_indices = np.searchsorted(classes, labels)
        offsets = rng.randint(1, num_classes, size=n)
        targets = classes[(source_indices + offsets) % num_classes]
        labels[mask] = targets[mask]

    return labels

def backdoor_attack(
    images,
    labels,
    target_label=0,
    poison_prob=0.4,
    trigger_value=1.0,
    seed=42
):

    labels = np.array(labels,dtype=np.int32).copy()
    rng = np.random.RandomState(seed)
    n = len(labels)

    if n == 0:
        return images, labels

    mask = rng.rand(n) < float(poison_prob)

    attacked_images = []
    for i, img in enumerate(images):
        if mask[i]:
            arr = img.numpy() if hasattr(img, "numpy") else np.array(img)
            h, w = arr.shape[:2]
            patch = max(2, min(h, w) // 12)

            arr[h - patch:h, w - patch:w, :] = trigger_value
            img = tf.convert_to_tensor(arr,dtype=tf.float32)
            labels[i] = int(target_label)

        attacked_images.append(img)

    return attacked_images, labels

def apply_attacks_for_client(
    images,
    labels,
    client_id,
    malicious_clients_attack_map,
    attack_params,
    seed=42
):

    attack_name = (malicious_clients_attack_map.get(client_id))
    if not attack_name:
        return images, labels

    if attack_name == "label_flip":
        params = attack_params.get("label_flip", {})

        attacked_labels = label_flip_attack(labels=labels,
                flip_prob=params.get("flip_prob",0.4),
                seed=params.get("seed", seed))

        return images, attacked_labels

    if attack_name == "backdoor":
        params = attack_params.get("backdoor",{})
        return backdoor_attack(images=images, labels=labels, target_label=params.get("target_label", 0),
            poison_prob=params.get("poison_prob", 0.4),
            trigger_value=params.get("trigger_value", 1.0),
            seed=params.get("seed", seed)
        )
    return images, labels

def assign_fake_clients(
    num_client,
    num_malicious,
    attack_list,
    seed=42,
    exclude_head=True,
    additional_malicious_clients=None,
    max_fake_ratio=0.7,
    attack_distribution=None
):

    rng = random.Random(seed)
    all_client_ids = [f"client_{i}" for i in range(num_client)]
    excluded = ({"client_0"} if exclude_head else set())

    candidates = [cid for cid in all_client_ids if cid not in excluded]
    additional = set(additional_malicious_clients or [])
    additional = {cid for cid in additional if cid in candidates}

    max_allowed = int(num_client * max_fake_ratio)

    k_total = min(num_malicious, len(candidates), max_allowed)
    k_remaining = max(0, k_total - len(additional))

    remaining_pool = [cid for cid in candidates if cid not in additional]
    sampled = rng.sample(remaining_pool, k=min(k_remaining, len(remaining_pool)))

    malicious_clients = sorted(list(additional.union(sampled)))
    malicious_clients_attack_map = {}

    if not malicious_clients:
        return malicious_clients, malicious_clients_attack_map, all_client_ids

    if attack_distribution is None:
        for cid in malicious_clients:
            malicious_clients_attack_map[cid] = rng.choice(attack_list)

        return malicious_clients, malicious_clients_attack_map, all_client_ids

    filtered = {
        attack: float(weight)
        for attack, weight in attack_distribution.items()
        if (attack in attack_list and float(weight) > 0)
    }

    if not filtered:
        raise ValueError("No valid attack distribution.")

    total_weight = sum(filtered.values())
    k = len(malicious_clients)
    expected = {
        attack: k * weight / total_weight
        for attack, weight in filtered.items()
    }

    counts = {
        attack: int(np.floor(value))
        for attack, value in expected.items()
    }

    remainder = (k - sum(counts.values()))
    frac_order = sorted(
        expected.keys(),
        key=lambda attack: (expected[attack] - counts[attack], expected[attack]),
        reverse=True
    )

    for i in range(remainder):
        counts[frac_order[i % len(frac_order)]] += 1

    assigned_attacks = []

    for attack in attack_list:
        assigned_attacks.extend([attack] * counts.get(attack,0))

    rng.shuffle(assigned_attacks)

    for cid, attack in zip(malicious_clients, assigned_attacks):
        malicious_clients_attack_map[cid] = attack

    return malicious_clients,malicious_clients_attack_map, all_client_ids
