import numpy as np

from src.config import constants


# ============================================================
# CLUSTERING SCORE
# ============================================================


def calculate_clustering_scores(
    hdbscan_results,
    head_client="client_0",
    noise_distance_threshold=constants.HDBSCAN_NOISE_DISTANCE_THRESHOLD,
):
    """
    Binary clustering score.

    Valid-cluster clients receive 1 only when they share client_0's cluster.
    A noise client (-1) is compared with client_0 in the same normalized-update
    PCA feature space used by HDBSCAN and receives 1 when its Euclidean
    distance is within ``noise_distance_threshold``. If client_0 is noise,
    the distance rule is used for every client in that round. Client_0 always
    receives 1.
    """

    if not 0 <= noise_distance_threshold <= 2:
        raise ValueError(
            "HDBSCAN noise distance threshold must be between 0 and 2."
        )

    clustering_scores = {}

    for round_key, result in hdbscan_results.items():
        clustering_scores[round_key] = {}

        labels = np.asarray(result["labels"])
        client_keys = result["client_keys"]
        features = np.asarray(result["features"])

        if head_client not in client_keys:
            raise ValueError(f"{head_client} missing in {round_key}")

        if len(labels) != len(client_keys) or len(features) != len(client_keys):
            raise ValueError(
                "HDBSCAN labels, features, and client keys must align "
                f"in {round_key}"
            )

        head_index = client_keys.index(head_client)
        head_cluster = labels[head_index]
        head_features = features[head_index]

        for client_index, (client_id, label) in enumerate(
            zip(client_keys, labels)
        ):
            if client_id == head_client:
                score = 1
            elif head_cluster == -1 or label == -1:
                distance_to_head = np.linalg.norm(
                    features[client_index] - head_features
                )
                score = int(distance_to_head <= noise_distance_threshold)
            else:
                score = int(label == head_cluster)

            clustering_scores[round_key][client_id] = score

    return clustering_scores


# ============================================================
# DIRECTION SCORE
# ============================================================


def calculate_direction_scores(head_angles, threshold):
    """
    Binary direction score.

    D_i = 1 if angle < threshold
    D_i = 0 otherwise
    """

    direction_scores = {}

    for round_key, clients in head_angles.items():

        direction_scores[round_key] = {}

        for client_id, angle in clients.items():

            if angle < threshold:

                score = 1

            else:

                score = 0

            direction_scores[round_key][client_id] = score

    return direction_scores


# ============================================================
# MAGNITUDE SCORE
# ============================================================


def calculate_magnitude_scores(
    magnitudes, head_client="client_0", threshold=0.25
):
    """
    Compare each raw delta magnitude with the head client's magnitude.

    With a threshold of 0.25, a client receives 1 when its magnitude is
    within the inclusive range [0.75 * head, 1.25 * head].
    Otherwise, it receives 0.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Magnitude threshold must be between 0 and 1.")

    magnitude_scores = {}

    for round_key, clients in magnitudes.items():

        magnitude_scores[round_key] = {}

        if head_client not in clients:
            raise ValueError(f"{head_client} missing in {round_key}")

        head_magnitude = float(clients[head_client])
        lower_bound = head_magnitude * (1 - threshold)
        upper_bound = head_magnitude * (1 + threshold)

        for client_id, magnitude in clients.items():
            value = float(magnitude)
            within_bounds = lower_bound <= value <= upper_bound
            on_boundary = np.isclose(value, lower_bound) or np.isclose(
                value, upper_bound
            )
            magnitude_scores[round_key][client_id] = int(
                within_bounds or on_boundary
            )

    return magnitude_scores


# ============================================================
# VALIDATION SCORE
# ============================================================


def calculate_validation_scores(
    validation_results, head_client="client_0", accuracy_tolerance=0.10
):
    """
    Binary validation score.

    Uses validation accuracy relative to the head client.

    V_i = 1 if:

        client_accuracy
        >=
        head_accuracy - accuracy_tolerance

    Otherwise:

        V_i = 0

    Example:

        head accuracy = 0.90
        tolerance     = 0.10

        minimum acceptable accuracy = 0.80
    """

    validation_scores = {}

    for round_key, clients in validation_results.items():

        validation_scores[round_key] = {}

        if head_client not in clients:

            raise ValueError(f"{head_client} missing in {round_key}")

        head_accuracy = float(clients[head_client]["accuracy"])

        minimum_accuracy = head_accuracy - accuracy_tolerance

        for client_id, result in clients.items():

            client_accuracy = float(result["accuracy"])

            if client_accuracy >= minimum_accuracy:

                score = 1

            else:

                score = 0

            validation_scores[round_key][client_id] = score

    return validation_scores


# ============================================================
# FINAL COMBINED TRUST SCORE
# ============================================================


def calculate_trust_scores(
    hdbscan_results, head_angles, magnitudes, validation_results, head_client="client_0"
):
    """
    Calculate the final combined trust score.

    Each component is binary:

        C_i ∈ {0,1}
        D_i ∈ {0,1}
        M_i ∈ {0,1}
        V_i ∈ {0,1}

    Final trust:

        T_i =
            w_c * C_i
            +
            w_d * D_i
            +
            w_m * M_i
            +
            w_v * V_i
    """

    # ========================================================
    # GET PARAMETERS FROM CONSTANTS
    # ========================================================

    w_c = constants.TRUST_WEIGHT_CLUSTER
    w_d = constants.TRUST_WEIGHT_DIRECTION
    w_m = constants.TRUST_WEIGHT_MAGNITUDE
    w_v = constants.TRUST_WEIGHT_VALIDATION

    direction_threshold = constants.DIRECTION_THRESHOLD

    magnitude_threshold = constants.MAGNITUDE_THRESHOLD

    validation_tolerance = constants.VALIDATION_ACCURACY_TOLERANCE

    noise_distance_threshold = constants.HDBSCAN_NOISE_DISTANCE_THRESHOLD

    # ========================================================
    # CHECK WEIGHTS
    # ========================================================

    total_weight = w_c + w_d + w_m + w_v

    if not np.isclose(total_weight, 1.0):

        raise ValueError(
            "Trust weights must sum to 1.0. " f"Current total = {total_weight}"
        )

    # ========================================================
    # CALCULATE INDIVIDUAL BINARY SCORES
    # ========================================================

    clustering_scores = calculate_clustering_scores(
        hdbscan_results,
        head_client=head_client,
        noise_distance_threshold=noise_distance_threshold,
    )

    direction_scores = calculate_direction_scores(
        head_angles, threshold=direction_threshold
    )

    magnitude_scores = calculate_magnitude_scores(
        magnitudes,
        head_client=head_client,
        threshold=magnitude_threshold,
    )

    validation_scores = calculate_validation_scores(
        validation_results,
        head_client=head_client,
        accuracy_tolerance=validation_tolerance,
    )

    # ========================================================
    # COMBINE SCORES
    # ========================================================

    trust_results = {}

    for round_key, cluster_clients in clustering_scores.items():

        trust_results[round_key] = {}

        for client_id in cluster_clients.keys():

            c = clustering_scores[round_key][client_id]

            d = direction_scores[round_key][client_id]

            m = magnitude_scores[round_key][client_id]

            v = validation_scores[round_key][client_id]

            # ------------------------------------------------
            # Weighted combined trust
            # ------------------------------------------------

            trust = w_c * c + w_d * d + w_m * m + w_v * v

            trust_results[round_key][client_id] = {
                "clustering_score": c,
                "direction_score": d,
                "magnitude_score": m,
                "validation_score": v,
                "trust_score": float(trust),
            }

    # ========================================================
    # RETURN
    # ========================================================

    return trust_results
