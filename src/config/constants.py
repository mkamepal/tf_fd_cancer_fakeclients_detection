# config/constants.py


# ============================================================
# FEDERATED LEARNING
# ============================================================

NUM_CLIENTS = 15

HEAD_CLIENT = "client_0"

FEDERATED_ROUNDS = 30

STORE_RESULTS = True


## Malicious client configurations

NUM_MALICIOUS = 9

MAX_FAKE_RATIO = 0.7

## Attack Configurations
LABEL_FLIP = True
FLIP_PROB = 0.4

BACKDOOR = True
BACKDOOR_POISON_PROB = 0.4
BACKDOOR_TARGET_LABEL = 0
BACKDOOR_TRIGGER_VALUE = 1.0

ATTACK_LIST = [
    "label_flip",
    "backdoor"
]

ATTACK_DISTRIBUTION = {
    "label_flip": 0.60,
    "backdoor": 0.40
}


# Reproducibility
SEED = 42


## Image Dataset Configurations

IMAGE_SIZE = (224, 224)
TARGET_SIZE = (64, 64)
TEST_SPLIT = 0.12
BATCH_SIZE = 8
MAX_SAMPLES = 10000

## Machine Learning Parameters
CLIENT_LEARNING_RATE = 0.01


## Trust Score weights and threshold

CLUSTERING_WEIGHT = 0.25
DIRECTION_WEIGHT = 0.25
MAGNITUDE_WEIGHT = 0.25
VALIDATION_WEIGHT = 0.25

# Require at least three of the four equally weighted defense checks.
TRUST_THRESHOLD = 0.50

## Data Dir path
DATA_DIR = '/home/mahesh/Desktop/Dissertation/tf_fd_cancer_fakeclients_detection/data/Kidney Cancer'

## results dir
RESULTS_DIR = "/home/mahesh/Desktop/Dissertation/tf_fd_cancer_fakeclients_detection/src/results"


TRUST_WEIGHT_CLUSTER = 0.25
TRUST_WEIGHT_DIRECTION = 0.25
TRUST_WEIGHT_MAGNITUDE = 0.25
TRUST_WEIGHT_VALIDATION = 0.25

# Keep the clustering space low-dimensional because each round contains only
# NUM_CLIENTS samples. This also makes the plotted PCA plane the space used by
# HDBSCAN rather than showing only a projection of a much larger space.
HDBSCAN_PCA_COMPONENTS = 2
HDBSCAN_MIN_CLUSTER_SIZE = 2

# Merge very close density subclusters in the normalized-update PCA space.
# This prevents a small benign population from being split into many pairs.
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.2
HDBSCAN_ALLOW_SINGLE_CLUSTER = True

# Maximum Euclidean distance from client_0 for an HDBSCAN noise point.
# Client updates are L2-normalized before PCA. PCA cannot increase pairwise
# Euclidean distance, so distances in the clustering space remain in [0, 2].
HDBSCAN_NOISE_DISTANCE_THRESHOLD = 1.5

# Clients with an angle strictly below this value are accepted.
DIRECTION_THRESHOLD = 90

# Maximum allowed relative magnitude difference from client_0.
# 0.25 accepts values from 75% through 125% of client_0's magnitude.
MAGNITUDE_THRESHOLD = 0.25

# Maximum allowed validation-accuracy drop from the head client.
# 0.06 = 6 percentage points
VALIDATION_ACCURACY_TOLERANCE = 0.06

USE_TRUST_FILTERING = True
