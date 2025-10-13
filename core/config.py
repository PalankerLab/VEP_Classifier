import os

# ==========================================
# Project Paths (anchored to the repo root,
# so scripts work regardless of the cwd)
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ==========================================
# Training Hyperparameters Configuration
# ==========================================
BATCHSIZE = 8
EPOCHS = 30
LR = 0.005
DROPOUT = 0.2
L2_LAMBDA = 0.0001


TMIN = 0
TMAX = 320
SNR_THRESHOLD = 1.0

RANDOM_STATE = 42

DEVICES = ["PRIMA_LE_DA", "MP20_LE_DA", "PRIMA_RCS_DA", "MP20_RCS_LA", "RB20_RCS_LA", "TEST_ALL"]
LABELS = ["BC_Only", "RGC_Only", "BC_and_RGC"]



