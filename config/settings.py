# Configuration settings
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_IMAGES_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Image settings
IMAGE_SIZE = (100, 100)  # Width, Height
NORMALIZE = True

# PCA settings
PCA_VARIANCE_THRESHOLD = 0.95
MIN_COMPONENTS = 2
MAX_COMPONENTS = 50

# Verification settings
THRESHOLD_MULTIPLIER = 1.5  # mean + multiplier*std
VERIFICATION_MODE = "euclidean"  # Options: "euclidean", "mahalanobis", "reconstruction"

# Visualization settings
SAVE_PLOTS = True
PLOT_FORMAT = "png"
DPI = 150
