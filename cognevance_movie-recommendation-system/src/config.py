"""Central configuration for the Movie Recommendation System."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "sample"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Cleaning
MIN_RATINGS_PER_MOVIE = 5     # drop movies with too little signal
MIN_RATINGS_PER_USER = 5      # drop users with too little signal

# Model
K_NEIGHBORS = 30              # number of similar users used for prediction
SHRINKAGE = 1.0               # damping so one lone neighbour can't dominate a score
TOP_N = 10                    # recommendations per user

# Evaluation
TEST_SIZE = 0.2
RELEVANT_THRESHOLD = 4.0      # a held-out rating >= 4 counts as "liked"
RANDOM_STATE = 42
