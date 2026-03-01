import os
import random
import numpy as np
import tensorflow as tf
from src.attack_details.attack_constants import (
    NUM_CLIENTS, NUM_MALICIOUS, FLIP_PROB, SEED, 
    IMAGE_SIZE, TARGET_SIZE, DATA_DIR
)

def load_image(path, img_size=IMAGE_SIZE):
    """Loads and normalizes an image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    return img


