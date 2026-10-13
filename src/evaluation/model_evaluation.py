from src.models.model_factory import create_keras_model
import tensorflow as tf


def evaluate_global_model(server_weights, test_dataset, model_builder=create_keras_model):
    with tf.device("/CPU:0"):
        model = model_builder()
        model.set_weights([
            w.numpy() if tf.is_tensor(w) else w
            for w in server_weights.trainable
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        loss, accuracy = model.evaluate(test_dataset, verbose=0)

    return {
        "loss": float(loss),
        "accuracy": float(accuracy)
    }
