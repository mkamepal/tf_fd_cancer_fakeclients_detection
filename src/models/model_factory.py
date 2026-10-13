"""Shared model selection for training, validation, and global evaluation."""

from src.config import constants


def create_keras_model():
    dataset = constants.get_dataset_name()
    if dataset == "cancer":
        from src.models.cancer_cnn import create_keras_model as build
    elif dataset == "mnist":
        from src.models.mnist_cnn import create_keras_model as build
    elif dataset == "emnist":
        from src.models.emnist_cnn import create_keras_model as build
    else:
        from src.models.cifar10_cnn import create_keras_model as build
    return build()
