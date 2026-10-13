"""Load MNIST from images, an NPZ file, or a ZIP, without split metadata."""

from pathlib import Path
import tempfile
import zipfile

import numpy as np
import tensorflow as tf

from src.config import constants


def _image_paths(root):
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)


def _load_source(source):
    if source.is_dir():
        paths = _image_paths(source)
        if paths:
            images, labels = [], []
            for path in paths:
                # Supports 0/image.png as well as train/0/image.png.
                if path.parent.name not in {str(i) for i in range(10)}:
                    raise ValueError(f"MNIST images must be in digit folders 0–9: {path}")
                image = tf.io.decode_image(tf.io.read_file(str(path)), channels=1,
                                           expand_animations=False)
                image = tf.image.resize(image, (28, 28))
                images.append(image.numpy())
                labels.append(int(path.parent.name))
            return np.stack(images), np.asarray(labels)
        files = sorted(source.rglob("*.npz"))
        if len(files) == 1:
            return _load_source(files[0])
        archives = sorted(source.glob("*.zip"))
        if not files and len(archives) == 1:
            return _load_source(archives[0])
        raise ValueError(f"Expected digit image folders, one NPZ, or one ZIP in {source}")
    if source.suffix.lower() == ".zip":
        # Validate the whole archive before extraction, including symlink entries.
        with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for item in archive.infolist():
                target = (root / item.filename).resolve()
                if root not in target.parents or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unsafe ZIP member: {item.filename}")
            archive.extractall(root)
            return _load_source(root)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if {"images", "labels"}.issubset(data.files):
                return data["images"], data["labels"]
            keys = {"x_train", "y_train", "x_test", "y_test"}
            if keys.issubset(data.files):
                # Treat all examples as one pool, just like the cancer folders.
                return (np.concatenate([data["x_train"], data["x_test"]]),
                        np.concatenate([data["y_train"], data["y_test"]]))
        raise ValueError("MNIST NPZ needs images/labels or x_train/y_train/x_test/y_test.")
    raise ValueError(f"Unsupported MNIST source: {source}; use images, .npz, or .zip.")


def load_mnist_arrays():
    root = Path(constants.MNIST_DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    if constants.MNIST_SOURCE is not None:
        source = Path(constants.MNIST_SOURCE)
        if not source.exists():
            raise FileNotFoundError(f"MNIST_SOURCE does not exist: {source}")
    elif _image_paths(root) or list(root.rglob("*.npz")) or list(root.glob("*.zip")):
        source = root
    else:
        source = Path(tf.keras.utils.get_file(
            "mnist.npz",
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            file_hash="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
            cache_dir=str(root.parent), cache_subdir=root.name,
        ))
    images, labels = _load_source(source)
    if images.ndim == 3:
        images = images[..., np.newaxis]
    labels = np.asarray(labels).reshape(-1)
    if (images.shape[1:] != (28, 28, 1) or len(images) != len(labels)
            or not len(labels) or not np.isfinite(images).all()
            or np.min(images) < 0 or np.max(images) > 255
            or not np.isin(labels, np.arange(10)).all()):
        raise ValueError("MNIST requires matching 28x28 grayscale images (0–255) and labels 0–9.")
    return images.astype(np.float32) / 255.0, labels.astype(np.int32)
