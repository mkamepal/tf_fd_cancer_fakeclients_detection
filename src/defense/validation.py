import tensorflow as tf

from src.models.model_factory import create_keras_model


def calculate_validation_results(
    captured_client_weights,
    validation_dataset,
    model_builder=create_keras_model,
):
    """Evaluate every client's model on the trusted validation dataset.

    The evaluation runs on the CPU and reuses one Keras model for all clients.
    This function only calculates and returns results. The experiment runner is
    responsible for storing them.
    """
    validation_results = {}

    with tf.device("/CPU:0"):
        evaluation_model = model_builder()
        evaluation_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        for internal_round_idx, round_client_weights in enumerate(
            captured_client_weights
        ):
            # run_tff normally passes one round, so this is round0weights.
            round_key = f"round{internal_round_idx}weights"
            validation_results[round_key] = {}

            print(f"\nEvaluating clients for {round_key}")

            for client_idx, client_weights in enumerate(round_client_weights):
                client_id = f"client_{client_idx}"
                evaluation_model.set_weights(
                    [
                        weight.numpy() if tf.is_tensor(weight) else weight
                        for weight in client_weights
                    ]
                )

                loss, accuracy = evaluation_model.evaluate(
                    validation_dataset,
                    verbose=0,
                )
                validation_results[round_key][client_id] = {
                    "loss": float(loss),
                    "accuracy": float(accuracy),
                }

                print(f"{client_id}: loss={loss:.4f}, " f"accuracy={accuracy:.4f}")

    return validation_results


def validation_accuracy_scores(validation_results):
    """Extract each client's validation accuracy from every round."""
    scores = {}

    for round_key, client_results in validation_results.items():
        scores[round_key] = {
            client_id: float(result["accuracy"])
            for client_id, result in client_results.items()
        }

    return scores
