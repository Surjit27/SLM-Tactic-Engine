"""Project-wide configuration for synthetic data + RL training."""

OBS_SIZE = 256
ACTION_SIZE = 32
TACTICAL_ACTIONS = 6

DATA_DIR = "data/generated"
RANDOM_SEED = 42

# Dataset defaults
DEFAULT_TRAIN_SAMPLES = 5000
DEFAULT_VAL_SAMPLES = 1000

# RL defaults
EPISODES = 200
STEPS_PER_EPISODE = 120
LEARNING_RATE = 0.01
GAMMA = 0.98
