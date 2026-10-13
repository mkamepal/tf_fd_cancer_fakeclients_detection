import gc
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow.keras.backend as K
import subprocess
from openpyxl import Workbook

from src.config import constants
from src.data_ops.client_partition import prepare_client_dataset
from src.defense.clustering import (
    calculate_hdbscan_results,
    save_all_hdbscan_plots,
)
from src.defense.direction import calculate_head_client_angles
from src.defense.magnitude import calculate_magnitudes
from src.defense.trustscore import calculate_trust_scores
from src.defense.validation import calculate_validation_results
from src.defense.weight_utils import calculate_client_deltas
from src.evaluation.model_evaluation import evaluate_global_model
from src.experiments.run_config import get_config
from src.federated.tff_process import build_federated_process


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def select_qualified_clients(round_trust_scores, threshold):
    """
    Select clients whose trust score is greater than
    or equal to the configured threshold.
    """
    selected_clients = []
    for client_id, result in round_trust_scores.items():
        if isinstance(result, dict):
            trust_score = float(result["trust_score"])
        else:
            trust_score = float(result)
        if trust_score >= threshold:
            selected_clients.append(client_id)
    return selected_clients


def aggregate_selected_clients(
    server_weights, client_trained_weights, selected_clients
):
    """
    Aggregate only the selected clients.
    If no client passes the trust threshold,
    keep the previous server model.
    """
    if len(selected_clients) == 0:
        print("\nWARNING: No client passed " "the trust threshold.")
        print("Keeping previous global model.")
        return server_weights
    # Convert client IDs to indices
    selected_indices = [int(client_id.split("_")[1]) for client_id in selected_clients]
    # Get selected client weights
    selected_weights = [client_trained_weights[index] for index in selected_indices]
    number_of_layers = len(selected_weights[0])
    aggregated_trainable = []
    # Average each layer
    for layer_idx in range(number_of_layers):
        layer_values = []
        for client_weights in selected_weights:
            weight = client_weights[layer_idx]
            if tf.is_tensor(weight):
                weight = weight.numpy()
            layer_values.append(tf.convert_to_tensor(weight))
        stacked_weights = tf.stack(layer_values, axis=0)
        mean_layer = tf.reduce_mean(stacked_weights, axis=0)
        aggregated_trainable.append(mean_layer)
    # Build new server weights
    new_server_weights = type(server_weights)(
        trainable=aggregated_trainable, non_trainable=server_weights.non_trainable
    )
    return new_server_weights


def aggregate_all_clients(server_weights, client_trained_weights):
    """
    Standard FedAvg-style equal-weight aggregation
    using all clients.
    """
    number_of_layers = len(client_trained_weights[0])
    aggregated_trainable = []
    for layer_idx in range(number_of_layers):
        layer_values = []
        for client_weights in client_trained_weights:
            weight = client_weights[layer_idx]
            if tf.is_tensor(weight):
                weight = weight.numpy()
            layer_values.append(tf.convert_to_tensor(weight))
        stacked_weights = tf.stack(layer_values, axis=0)
        mean_layer = tf.reduce_mean(stacked_weights, axis=0)
        aggregated_trainable.append(mean_layer)
    new_server_weights = type(server_weights)(
        trainable=aggregated_trainable, non_trainable=server_weights.non_trainable
    )
    return new_server_weights


##TensorFlow Federated is starting separate worker_binary processes, and those worker processes are not shutting down when your main experiment finishes.
def cleanup_experiment():
    print("\nCleaning experiment resources...")

    try:
        result = subprocess.run(
            [
                "pkill",
                "-TERM",
                "-u",
                str(os.getuid()),
                "-f",
                r"tensorflow_federated.*worker_binary",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print("TFF worker processes terminated.")
        elif result.returncode == 1:
            print("No remaining TFF worker processes found.")
        else:
            print(
                "TFF worker cleanup warning: "
                f"pkill returned {result.returncode}: {result.stderr.strip()}"
            )

    except (OSError, subprocess.SubprocessError) as error:
        print(f"TFF worker cleanup warning: {error}")

    K.clear_session()
    gc.collect()
    print("Cleanup completed.")

def aggregate_client_metrics(client_metrics, selected_clients):
    """Calculate example-weighted metrics for only the selected clients."""
    selected_indices = [int(client_id.split("_")[1]) for client_id in selected_clients]

    loss_sum = sum(float(client_metrics[index]["loss_sum"]) for index in selected_indices)
    correct_sum = sum(
        float(client_metrics[index]["correct_sum"]) for index in selected_indices
    )
    num_examples = sum(
        float(client_metrics[index]["num_examples"]) for index in selected_indices
    )

    if num_examples == 0:
        return {"train_loss": float("nan"), "train_accuracy": float("nan")}

    return {
        "train_loss": loss_sum / num_examples,
        "train_accuracy": correct_sum / num_examples,
    }


def _round_number(round_key):
    return int(round_key.removeprefix("round").removesuffix("weights")) + 1


def _client_ids(client_results):
    return sorted(client_results, key=lambda client_id: int(client_id.split("_")[1]))


def _set_column_widths(worksheet, widths):
    for column, width in widths.items():
        worksheet.column_dimensions[column].width = width


def _next_available_file_path(file_path):
    """Return a new path without overwriting an earlier experiment."""
    if not os.path.exists(file_path):
        return file_path

    base_path, extension = os.path.splitext(file_path)
    run_number = 2
    candidate = f"{base_path}_{run_number}{extension}"
    while os.path.exists(candidate):
        run_number += 1
        candidate = f"{base_path}_{run_number}{extension}"
    return candidate


def _save_results_file(file_path, sheet_name, headers, rows, widths):
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    worksheet.append(headers)
    for row in rows:
        worksheet.append(row)
    _set_column_widths(worksheet, widths)
    workbook.save(file_path)


def save_experiment_results_to_excel(
    round_losses,
    round_accuracies,
    global_model_losses,
    global_model_accuracies,
    all_magnitudes,
    all_head_angles,
    all_hdbscan_results,
    all_validation_results,
    all_trust_scores,
    all_selected_clients,
    file_paths=None,
):
    """Save global metrics and, when filtering is enabled, defense results."""
    iid_name = "iid_" + "_".join(constants.get_iid_techniques())
    output_dir = os.path.join(constants.RESULTS_DIR, iid_name)
    os.makedirs(output_dir, exist_ok=True)
    if file_paths is None:
        trust_status = str(constants.USE_TRUST_FILTERING).lower()
        suffix = (
            f"clients_{constants.NUM_CLIENTS}"
            f"_malicious_{constants.NUM_MALICIOUS}"
            f"_rounds_{constants.FEDERATED_ROUNDS}"
            f"_trust_is_{trust_status}.xlsx"
        )
        result_file_prefixes = {"round_metrics": "round_metrics"}
        if constants.USE_TRUST_FILTERING:
            result_file_prefixes.update(
                {
                    "magnitudes": "magnitudes",
                    "directions": "directions",
                    "hdbscan": "hdbscan",
                    "validation": "validation",
                    "trust_scores": "trust_scores",
                    "selected_clients": "selected",
                }
            )
        file_paths = {
            result_type: _next_available_file_path(
                os.path.join(
                    output_dir,
                    (
                        f"globalmodel_results_{constants.NUM_MALICIOUS}"
                        f"_trust_is_{trust_status}.xlsx"
                        if result_type == "round_metrics"
                        else f"{file_prefix}_{suffix}"
                    ),
                )
            )
            for result_type, file_prefix in result_file_prefixes.items()
        }

    metric_rows = [
        [
            round_idx,
            float(train_loss),
            float(train_accuracy),
            float(global_loss),
            float(global_accuracy),
        ]
        for round_idx, (
            train_loss,
            train_accuracy,
            global_loss,
            global_accuracy,
        ) in enumerate(
            zip(
                round_losses,
                round_accuracies,
                global_model_losses,
                global_model_accuracies,
            ),
            start=1,
        )
    ]
    _save_results_file(
        file_paths["round_metrics"],
        "Round Metrics",
        [
            "Round",
            "Training Loss",
            "Training Accuracy",
            "Global Model Test Loss",
            "Global Model Test Accuracy",
        ],
        metric_rows,
        {"A": 12, "B": 20, "C": 22, "D": 25, "E": 29},
    )

    if not constants.USE_TRUST_FILTERING:
        print("\nExperiment result file saved to:")
        print(file_paths["round_metrics"])
        return file_paths

    magnitude_rows = []
    for round_key, clients in all_magnitudes.items():
        for client_id in _client_ids(clients):
            magnitude_rows.append(
                [_round_number(round_key), client_id, float(clients[client_id])]
            )
    _save_results_file(
        file_paths["magnitudes"],
        "Magnitudes",
        ["Round", "Client ID", "Magnitude"],
        magnitude_rows,
        {"A": 12, "B": 18, "C": 22},
    )

    direction_rows = []
    for round_key, clients in all_head_angles.items():
        for client_id in _client_ids(clients):
            direction_rows.append(
                [
                    _round_number(round_key),
                    "client_0",
                    client_id,
                    float(clients[client_id]),
                ]
            )
    _save_results_file(
        file_paths["directions"],
        "Directions",
        ["Round", "Head Client", "Client ID", "Angle (Degrees)"],
        direction_rows,
        {"A": 12, "B": 18, "C": 18, "D": 22},
    )

    hdbscan_rows = []
    for round_key, result in all_hdbscan_results.items():
        for client_id, label, probability in zip(
            result["client_keys"], result["labels"], result["probabilities"]
        ):
            hdbscan_rows.append(
                [
                    _round_number(round_key),
                    client_id,
                    int(label),
                    float(probability),
                ]
            )
    _save_results_file(
        file_paths["hdbscan"],
        "HDBSCAN",
        ["Round", "Client ID", "Cluster Label", "HDBSCAN Probability"],
        hdbscan_rows,
        {"A": 12, "B": 18, "C": 18, "D": 25},
    )

    validation_rows = []
    for round_key, clients in all_validation_results.items():
        for client_id in _client_ids(clients):
            result = clients[client_id]
            validation_rows.append(
                [
                    _round_number(round_key),
                    client_id,
                    float(result["loss"]),
                    float(result["accuracy"]),
                ]
            )
    _save_results_file(
        file_paths["validation"],
        "Validation",
        ["Round", "Client ID", "Validation Loss", "Validation Accuracy"],
        validation_rows,
        {"A": 12, "B": 18, "C": 22, "D": 24},
    )

    trust_rows = []
    trust_headers = [
        "Round",
        "Client ID",
        "Clustering Score",
        "Direction Score",
        "Magnitude Score",
        "Validation Score",
        "Trust Score",
    ]
    for round_key, clients in all_trust_scores.items():
        for client_id in _client_ids(clients):
            result = clients[client_id]
            trust_rows.append(
                [
                    _round_number(round_key),
                    client_id,
                    int(result["clustering_score"]),
                    int(result["direction_score"]),
                    int(result["magnitude_score"]),
                    int(result["validation_score"]),
                    float(result["trust_score"]),
                ]
            )
    _save_results_file(
        file_paths["trust_scores"],
        "Trust Scores",
        trust_headers,
        trust_rows,
        {"A": 12, "B": 18, "C": 20, "D": 20, "E": 20, "F": 20, "G": 18},
    )

    selected_rows = [
        [_round_number(round_key), client_id]
        for round_key, selected_clients in all_selected_clients.items()
        for client_id in _client_ids(selected_clients)
    ]
    _save_results_file(
        file_paths["selected_clients"],
        "Selected Clients",
        ["Round", "Client ID"],
        selected_rows,
        {"A": 12, "B": 18},
    )

    print("\nExperiment result files saved to:")
    for path in file_paths.values():
        print(path)
    return file_paths


def main():
    K.clear_session()
    gc.collect()
    config = get_config()
    print("\nIID techniques:", ", ".join(constants.get_iid_techniques()))
    print("\nDataset:", constants.get_dataset_name())
    set_seeds(config.seed)
    print("\nTrust filtering enabled:", constants.USE_TRUST_FILTERING)
    attack_params = {
        "label_flip": {"flip_prob": constants.FLIP_PROB, "seed": config.seed},
        "backdoor": {
            "target_label": constants.BACKDOOR_TARGET_LABEL,
            "poison_prob": constants.BACKDOOR_POISON_PROB,
            "trigger_value": constants.BACKDOOR_TRIGGER_VALUE,
            "seed": config.seed,
        },
    }
    (client_datasets, client_test_datasets, malicious_clients, malicious_attack_map) = (
        prepare_client_dataset(
            data_dir=config.data_dir,
            num_client=config.num_clients,
            num_malicious=config.num_malicious,
            attack_list=constants.ATTACK_LIST,
            attack_params=attack_params,
            seed=config.seed,
            max_samples=config.max_samples,
            attack_distribution=constants.ATTACK_DISTRIBUTION,
            test_split=config.test_split,
            batch_size=config.batch_size,
        )
    )
    print("\nMalicious clients:")
    print(malicious_clients)
    print("\nAttack map:")
    print(malicious_attack_map)
    # We are using client_0 test data as the trusted
    # validation dataset.
    #
    # Later create a separate validation partition.
    head_validation_dataset = client_test_datasets["client_0"]
    input_spec = client_datasets["client_0"].element_spec
    (initialize_fn, next_fn) = build_federated_process(
        input_spec=input_spec, learning_rate=config.learning_rate
    )
    server_weights = initialize_fn()
    client_datasets_list = [
        client_datasets[f"client_{i}"] for i in range(config.num_clients)
    ]
    round_losses = []
    round_accuracies = []
    global_model_losses = []
    global_model_accuracies = []
    all_magnitudes = {}
    all_head_angles = {}
    all_hdbscan_results = {}
    all_validation_results = {}
    all_trust_scores = {}
    all_selected_clients = {}
    results_file_paths = None
    for round_idx in range(config.rounds):
        print("\n" + "=" * 70)
        print(f"ROUND {round_idx + 1}")
        print("=" * 70)
        # CURRENT GLOBAL MODEL W_t
        server_weights_before_round = server_weights
        print("\nTraining clients...")
        client_trained_weights, client_metrics, round_metrics = next_fn(
            server_weights, client_datasets_list
        )
        if constants.USE_TRUST_FILTERING:
            print("\nTrust-based filtering ENABLED.")
            # Defense functions expect a round list.
            # We process one round at a time.
            current_round_client_weights = [client_trained_weights]
            # A client update is measured from the model that the server sent
            # at the start of this round: delta_i = local_model_i - W_t.
            # Using the all-client mean here measures a centered residual
            # instead. That reference is contaminated by malicious clients and
            # caused their residual magnitudes to converge toward client_0's
            # magnitude in later rounds.
            current_round_server_weights = [server_weights_before_round]
            print("\nCalculating client deltas...")
            client_deltas = calculate_client_deltas(
                captured_client_weights=current_round_client_weights,
                server_weights_before_round=current_round_server_weights,
            )
            print("\nCalculating delta magnitudes...")
            magnitudes = calculate_magnitudes(client_deltas)
            print("\nCalculating head-client angles...")
            # Match fakeclients_segregation.ipynb's direction section. The
            # notebook reassigns client_weights = client_weight_deltas, where
            # those deltas are centered on the post-round all-client FedAvg
            # model. Keep this reference direction-only so magnitude and
            # HDBSCAN continue using updates measured from W_t.
            direction_reference_server = aggregate_all_clients(
                server_weights=server_weights_before_round,
                client_trained_weights=client_trained_weights,
            )
            direction_client_deltas = calculate_client_deltas(
                captured_client_weights=current_round_client_weights,
                server_weights_before_round=[direction_reference_server],
            )
            head_angles = calculate_head_client_angles(
                direction_client_deltas, head_client="client_0"
            )
            print("\nRunning HDBSCAN...")
            hdbscan_results = calculate_hdbscan_results(
                client_weight_deltas=client_deltas
            )
            print("\nEvaluating clients " "using trusted validation data...")
            validation_results = calculate_validation_results(
                captured_client_weights=current_round_client_weights,
                validation_dataset=head_validation_dataset,
            )
            print("\nCalculating trust scores...")
            trust_scores = calculate_trust_scores(
                hdbscan_results=hdbscan_results,
                head_angles=head_angles,
                magnitudes=magnitudes,
                validation_results=validation_results,
                head_client="client_0",
            )
            # Since the defense functions receive only one
            # round, their internal round name is always:
            #
            # round0weights
            temporary_round_key = "round0weights"
            actual_round_key = f"round{round_idx}weights"
            all_magnitudes[actual_round_key] = magnitudes[temporary_round_key]
            all_head_angles[actual_round_key] = head_angles[temporary_round_key]
            all_hdbscan_results[actual_round_key] = hdbscan_results[temporary_round_key]
            all_validation_results[actual_round_key] = validation_results[
                temporary_round_key
            ]
            all_trust_scores[actual_round_key] = trust_scores[temporary_round_key]
            current_round_trust_scores = trust_scores[temporary_round_key]
            selected_clients = select_qualified_clients(
                round_trust_scores=current_round_trust_scores,
                threshold=constants.TRUST_THRESHOLD,
            )
            all_selected_clients[actual_round_key] = selected_clients
            print("\nClient Trust Decisions")
            print("-" * 70)
            client_ids = sorted(
                current_round_trust_scores.keys(),
                key=lambda value: int(value.split("_")[1]),
            )
            for client_id in client_ids:
                result = current_round_trust_scores[client_id]
                if isinstance(result, dict):
                    trust_value = float(result["trust_score"])
                else:
                    trust_value = float(result)
                if client_id in selected_clients:
                    decision = "QUALIFIED"
                else:
                    decision = "REJECTED"
                print(
                    f"{client_id:12s} "
                    f"Trust = "
                    f"{trust_value:.4f} "
                    f"-> "
                    f"{decision}"
                )
            print("\nSelected clients:")
            print(selected_clients)
            print("\nAggregating qualified clients...")
            server_weights = aggregate_selected_clients(
                server_weights=server_weights_before_round,
                client_trained_weights=client_trained_weights,
                selected_clients=selected_clients,
            )
            reported_round_metrics = aggregate_client_metrics(
                client_metrics=client_metrics,
                selected_clients=selected_clients,
            )
            print(
                f"\nQualified clients: "
                f"{len(selected_clients)}"
                f"/"
                f"{config.num_clients}"
            )
            del client_deltas
            del direction_client_deltas
            del direction_reference_server
            del magnitudes
            del head_angles
            del hdbscan_results
            del validation_results
            del trust_scores
            gc.collect()
        else:
            print("\nTrust-based filtering DISABLED.")
            print("Using standard aggregation " "with all clients.")
            server_weights = aggregate_all_clients(
                server_weights=server_weights_before_round,
                client_trained_weights=client_trained_weights,
            )
            # With filtering disabled, every client is selected, so the round
            # metrics already returned by TFF are the correct aggregate.
            reported_round_metrics = round_metrics

        loss_value = float(reported_round_metrics["train_loss"])
        accuracy_value = float(reported_round_metrics["train_accuracy"])
        round_losses.append(loss_value)
        round_accuracies.append(accuracy_value)
        print("\nRound results for aggregated clients:")
        print(f"Loss: {loss_value:.4f}")
        print(f"Accuracy: {accuracy_value:.4f}")

        global_results = evaluate_global_model(
            server_weights=server_weights,
            test_dataset=client_test_datasets["client_0"],
        )
        global_loss = float(global_results["loss"])
        global_accuracy = float(global_results["accuracy"])
        global_model_losses.append(global_loss)
        global_model_accuracies.append(global_accuracy)
        print("\nGlobal model test results:")
        print(f"Global Model Loss: {global_loss:.4f}")
        print(f"Global Model Accuracy: {global_accuracy:.4f}")
        print(f"\nRound " f"{round_idx + 1} " f"completed.")
        results_file_paths = save_experiment_results_to_excel(
            round_losses=round_losses,
            round_accuracies=round_accuracies,
            global_model_losses=global_model_losses,
            global_model_accuracies=global_model_accuracies,
            all_magnitudes=all_magnitudes,
            all_head_angles=all_head_angles,
            all_hdbscan_results=all_hdbscan_results,
            all_validation_results=all_validation_results,
            all_trust_scores=all_trust_scores,
            all_selected_clients=all_selected_clients,
            file_paths=results_file_paths,
        )

        if all_hdbscan_results:
            hdbscan_file_path = results_file_paths["hdbscan"]
            plot_output_dir = f"{os.path.splitext(hdbscan_file_path)[0]}_plots"
            save_all_hdbscan_plots(
                hdbscan_results=all_hdbscan_results,
                output_dir=plot_output_dir,
                show_plots=False,
            )

    print("\n" + "=" * 70)
    print("FEDERATED TRAINING COMPLETED")
    print("=" * 70)
    print("\nEvaluating final global model...")
    final_results = evaluate_global_model(
        server_weights=server_weights, test_dataset=client_test_datasets["client_0"]
    )
    print("\nFinal Global Model")
    print(final_results)
    # Print Filtering techniques
    if constants.USE_TRUST_FILTERING:
        print("\n" + "=" * 70)
        print("CLIENT SELECTION SUMMARY")
        print("=" * 70)
        for round_idx in range(config.rounds):
            round_key = f"round{round_idx}weights"
            selected_clients = all_selected_clients[round_key]
            print(
                f"Round {round_idx + 1}: {len(selected_clients)} clients selected -> {selected_clients}"
            )
    print("\nExperiment completed.")


if __name__ == "__main__":
    K.clear_session()
    gc.collect()

    tff.backends.native.set_sync_local_cpp_execution_context(
        max_concurrent_computation_calls=4
    )
    try:
        main()
    finally:
        cleanup_experiment()
