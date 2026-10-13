"""Offline checks: run with python -m unittest discover -s tests -v."""

import unittest
from pathlib import Path
import tempfile
import zipfile
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from src.config import constants
from src.data_ops.attacks import label_flip_attack
from src.data_ops.client_partition import prepare_client_dataset
from src.federated.tff_process import build_federated_process
from src.models.model_factory import create_keras_model


def dataset_flags(name):
    return patch.multiple(
        constants, DATASET_CANCER=name == "cancer",
        DATASET_MNIST=name == "mnist",
        DATASET_EMNIST=name == "emnist", DATASET_CIFAR10=name == "cifar10",
    )


class DatasetTests(unittest.TestCase):
    def test_exactly_one_boolean_required(self):
        for flags in [(False, False, False), (True, True, False), (1, False, False)]:
            with patch.multiple(constants, DATASET_CANCER=flags[0],
                                DATASET_MNIST=False,
                                DATASET_EMNIST=flags[1], DATASET_CIFAR10=flags[2]):
                with self.assertRaises(ValueError):
                    constants.get_dataset_name()

    def test_models_train_and_convert_to_tff(self):
        for name, shape, classes in [
            ("cancer", (64, 64, 3), 2),
            ("mnist", (28, 28, 1), 10),
            ("emnist", (28, 28, 1), 47),
            ("cifar10", (32, 32, 3), 10),
        ]:
            with self.subTest(dataset=name), dataset_flags(name):
                model = create_keras_model()
                self.assertEqual(model.input_shape[1:], shape)
                self.assertEqual(model.output_shape[-1], classes)
                self.assertFalse(model.non_trainable_weights)
                x = tf.ones((2,) + shape)
                y = tf.constant([0, classes - 1], dtype=tf.int32)
                model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy")
                self.assertTrue(np.isfinite(model.train_on_batch(x, y)))
                spec = (tf.TensorSpec((None,) + shape, tf.float32),
                        tf.TensorSpec((None,), tf.int32))
                initialize, next_round = build_federated_process(spec)
                self.assertIsNotNone(initialize.type_signature)
                self.assertIsNotNone(next_round.type_signature)

    def test_benchmark_loading_splitting_and_clean_test_data(self):
        for name, shape in [("mnist", (28, 28, 1)), ("emnist", (28, 28, 1)), ("cifar10", (32, 32, 3))]:
            images = np.random.RandomState(42).randint(0, 256, (24,) + shape, dtype=np.uint8)
            labels = np.arange(24, dtype=np.int64) % 10
            target = ("src.data_ops.mnist_data.load_mnist_arrays" if name == "mnist"
                      else "src.data_ops.benchmark_data._load_tfds_arrays")
            with self.subTest(dataset=name), dataset_flags(name), patch(
                target, return_value=(images.astype(np.float32) / 255, labels.astype(np.int32))
            ) as load:
                train, test, malicious, attacks = prepare_client_dataset(
                    data_dir="unused", num_client=3, num_malicious=1,
                    attack_list=["backdoor"],
                    attack_params={"backdoor": {"poison_prob": 1.0, "target_label": 0}},
                    max_samples=24, test_split=0.25, batch_size=2,
                )
                self.assertNotIn("client_0", malicious)
                self.assertEqual(len(train), 3)
                for client in train:
                    x, y = next(iter(train[client]))
                    self.assertEqual(tuple(x.shape[1:]), shape)
                    self.assertEqual(y.dtype, tf.int32)
                    self.assertGreaterEqual(float(tf.reduce_min(x)), 0.0)
                    self.assertLessEqual(float(tf.reduce_max(x)), 1.0)
                    self.assertEqual(sum(len(y) for _, y in train[client]), 6)
                    self.assertEqual(sum(len(y) for _, y in test[client]), 2)
                # Recreate the same splits without poisoning; test data must match.
                clean_train, clean_test, _, _ = prepare_client_dataset(
                    data_dir="unused", num_client=3, num_malicious=0,
                    attack_list=["backdoor"], attack_params={},
                    max_samples=24, test_split=0.25, batch_size=2,
                )
                for client in test:
                    for (x, y), (cx, cy) in zip(test[client], clean_test[client]):
                        np.testing.assert_array_equal(x, cx)
                        np.testing.assert_array_equal(y, cy)
                for x, y in train[malicious[0]]:
                    np.testing.assert_array_equal(y, np.zeros(len(y), dtype=np.int32))
                    np.testing.assert_array_equal(x[:, -2:, -2:, :], 1.0)

    def test_label_flip_with_non_contiguous_classes(self):
        labels = np.array([12, 27, 46] * 10, dtype=np.int32)
        flipped = label_flip_attack(labels, flip_prob=1.0)
        self.assertTrue(np.all(flipped != labels))
        self.assertTrue(np.isin(flipped, [12, 27, 46]).all())
        np.testing.assert_array_equal(label_flip_attack(labels, 0.0), labels)

    def test_mnist_npz_zip_images_and_download(self):
        from src.data_ops.mnist_data import load_mnist_arrays
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = dict(x_train=np.zeros((3, 28, 28), dtype=np.uint8),
                        y_train=np.array([0, 1, 2]),
                        x_test=np.full((2, 28, 28), 255, dtype=np.uint8),
                        y_test=np.array([8, 9]))
            npz = root / "mnist.npz"
            np.savez(npz, **data)
            archive = root / "digits.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.write(npz, "nested/mnist.npz")
            for source in (npz, archive):
                with patch.multiple(constants, MNIST_DATA_DIR=folder, MNIST_SOURCE=str(source)):
                    x, y = load_mnist_arrays()
                    self.assertEqual(x.shape, (5, 28, 28, 1))
                    np.testing.assert_array_equal(y, [0, 1, 2, 8, 9])
                    self.assertEqual(float(x[-1].max()), 1.0)
            image_root = root / "images"
            (image_root / "7").mkdir(parents=True)
            tf.io.write_file(str(image_root / "7" / "digit.png"),
                             tf.io.encode_png(np.full((28, 28, 1), 255, dtype=np.uint8)))
            with patch.multiple(constants, MNIST_DATA_DIR=folder, MNIST_SOURCE=str(image_root)):
                x, y = load_mnist_arrays()
                np.testing.assert_array_equal(y, [7])
                self.assertEqual(float(x.max()), 1.0)
            with patch.multiple(constants, MNIST_DATA_DIR=str(root / "download"), MNIST_SOURCE=None), patch(
                "src.data_ops.mnist_data.tf.keras.utils.get_file", return_value=str(npz)
            ) as download:
                self.assertEqual(len(load_mnist_arrays()[1]), 5)
                download.assert_called_once()
            with zipfile.ZipFile(root / "bad.zip", "w") as z:
                z.writestr("../escape.png", b"invalid")
            with patch.multiple(constants, MNIST_DATA_DIR=folder, MNIST_SOURCE=str(root / "bad.zip")):
                with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
                    load_mnist_arrays()


if __name__ == "__main__":
    unittest.main()
