import collections
import tensorflow as tf
import tensorflow_federated as tff

from src.models.model_factory import create_keras_model


def build_tff_model(input_spec):

    keras_model = create_keras_model()

    return tff.learning.models.functional_model_from_keras(
        keras_model,
        input_spec=input_spec,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics_constructor=collections.OrderedDict(
            accuracy=tf.keras.metrics.SparseCategoricalAccuracy
        )
    )


def build_federated_process(input_spec, learning_rate=0.01):

    tff_model = build_tff_model(input_spec)


    # ---------------------------------------------------------
    # SERVER INITIALIZATION
    # ---------------------------------------------------------

    @tff.tensorflow.computation
    def server_init():
        return tff.learning.models.ModelWeights(
            *tff_model.initial_weights
        )

    # Type of the complete model weights
    model_weights_type = server_init.type_signature.result
    # Type of one client's TensorFlow dataset
    tf_dataset_type = tff.SequenceType(tff.types.tensorflow_to_type(input_spec))


    # ---------------------------------------------------------
    # FEDERATED INITIALIZATION
    # ---------------------------------------------------------

    @tff.federated_computation
    def initialize_fn():

        return tff.federated_eval(
            server_init,
            tff.SERVER
        )


    # ---------------------------------------------------------
    # CLIENT LOCAL TRAINING
    # ---------------------------------------------------------

    @tf.function
    def client_update(
        model,
        dataset,
        initial_weights,
        client_optimizer
    ):

        # Get trainable weights received from server
        client_weights =  initial_weights.trainable
        # Initialize optimizer state
        optimizer_state = client_optimizer.initialize(tf.nest.map_structure(tf.TensorSpec.from_tensor,client_weights))

        # Metrics
        total_loss = tf.constant(0.0,dtype=tf.float32)
        total_correct = tf.constant(0.0,dtype=tf.float32)
        total_examples = tf.constant(0.0,dtype=tf.float32)


        # -----------------------------------------------------
        # LOCAL CLIENT TRAINING LOOP
        # -----------------------------------------------------

        for batch in dataset:
            x, y = batch
            with tf.GradientTape() as tape:
                tape.watch(client_weights)
                # Forward pass
                outputs = model.predict_on_batch(model_weights=(client_weights,()),
                                                 x=x,
                                                 training=True
                                                 )
                # Calculate loss
                loss = model.loss(output=outputs, label=y)
            # Calculate gradients
            grads = tape.gradient(loss, client_weights)
            # Update client weights
            (optimizer_state, client_weights) = client_optimizer.next(optimizer_state,
                                                                      weights=client_weights,
                                                                      gradients=grads
                                                                      )
            # Number of examples in this batch
            batch_size = tf.cast(tf.shape(y)[0], tf.float32)
            # Predicted class
            preds = tf.argmax(outputs,
                              axis=1,
                              output_type=tf.int32
                              )
            # Number of correct predictions
            correct = tf.reduce_sum(tf.cast(tf.equal(preds, y), tf.float32))
            # Accumulate metrics
            total_loss += (loss * batch_size)
            total_correct += correct
            total_examples += batch_size
        # Client metrics
        metrics = collections.OrderedDict(loss_sum=total_loss,
                                          correct_sum=total_correct,
                                          num_examples=total_examples
                                          )
        # Return locally trained weights + metrics
        return client_weights, metrics


    # ---------------------------------------------------------
    # CLIENT TENSORFLOW COMPUTATION
    # ---------------------------------------------------------

    @tff.tensorflow.computation(
        tf_dataset_type,
        model_weights_type
    )
    def client_update_fn(
        tf_dataset,
        server_weights
    ):

        optimizer = tff.learning.optimizers.build_sgdm(learning_rate=learning_rate)
        return client_update(tff_model, tf_dataset,server_weights,optimizer)


    # ---------------------------------------------------------
    # FINALIZE ROUND METRICS
    # ---------------------------------------------------------

    @tff.tensorflow.computation(
        collections.OrderedDict(
            loss_sum=tf.float32,
            correct_sum=tf.float32,
            num_examples=tf.float32
        )
    )
    def finalize_round_metrics(
        metric_sums
    ):
        return collections.OrderedDict(
            train_loss=(metric_sums["loss_sum"]/metric_sums["num_examples"]),
            train_accuracy=(metric_sums["correct_sum"]/metric_sums["num_examples"])
        )
    # ---------------------------------------------------------
    # FEDERATED TYPES
    # ---------------------------------------------------------

    federated_server_type = tff.FederatedType(model_weights_type,tff.SERVER)
    federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)
    # ---------------------------------------------------------
    # ONE FEDERATED TRAINING ROUND
    # ---------------------------------------------------------

    @tff.federated_computation(
        federated_server_type,
        federated_dataset_type
    )
    def next_fn(server_weights, federated_dataset):

        server_weights_at_client = (tff.federated_broadcast(server_weights))

        #Traning each client locally
        client_outputs = (tff.federated_map(client_update_fn,(federated_dataset, server_weights_at_client)))
        client_trainable_weights = (client_outputs[0])
        client_metrics = (client_outputs[1])
        metric_sums = (tff.federated_sum(client_metrics))

        round_metrics = (tff.federated_map(finalize_round_metrics,metric_sums))
        return client_trainable_weights, client_metrics, round_metrics

    return initialize_fn, next_fn
