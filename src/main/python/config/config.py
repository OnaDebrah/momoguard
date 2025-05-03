# Configuration constants
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Model parameters
RANDOM_STATE = 42
FRAUD_RATIO = 0.05
TEST_SIZE = 0.25