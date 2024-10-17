# config.py
# %%writefile config.py

# Try to import torch, but don't fail if it's not available
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEVICE = 'cpu'


# Data configuration
PHISHING_URLS_FILE = 'phishing_urls.json'
LEGITIMATE_URLS_FILE = 'legitimate_urls.json'
TEST_SPLIT = 0.2

# Feature extraction configuration
NUM_FEATURES = 14

# DQN model configuration
STATE_SIZE = NUM_FEATURES
ACTION_SIZE = 2  # Binary classification: phishing or legitimate

# Neural network configuration
HIDDEN_LAYER_1_UNITS = 64
HIDDEN_LAYER_2_UNITS = 32

# Training configuration
EPISODES = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Memory configuration
MEMORY_SIZE = 10000

# Environment configuration
REWARD_CORRECT = 1
REWARD_INCORRECT = -1

# Evaluation configuration
EVALUATION_EPISODES = 100

SAVE_INTERVAL = 100

# File paths
MODEL_SAVE_PATH = 'phishing_detection_model.h5'
RESULTS_SAVE_PATH = 'evaluation_results.json'
TEST_PREDICTIONS_SAVE_PATH = 'test_predictions_results.json'

# Random seed for reproducibility
RANDOM_SEED = 42

# Device configuration (for PyTorch)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Logging configuration
LOG_INTERVAL = 100  # Log every 100 episodes during training