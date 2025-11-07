import pickle
from datetime import datetime
BASE_DIR = "/home/m/man0302/CS4243/project/Handwriting-Synthesis"
with open(f"{BASE_DIR}/src/ctoi.txt", "rb") as file:
    enc_dict = pickle.load(file)

SRC_DIR = f"/home/m/man0302/CS4243/project/CS4243-CAPTCHA/Datasets/full/train"
OUT_DIR = f"{BASE_DIR}/src/out"
CHECKPOINT_PATH
RUNS_DIR = f"{BASE_DIR}/src/runs"
BATCH_SIZE = 64
LOG=datetime.now().strftime("%Y%m%d-%H%M")
NUM_TOKENS = len(enc_dict)
EMBEDDING_SIZE = 128
NUM_LAYERS = 4  # Ideally 4
PADDING_IDX = 0
Z_LEN = 128  # Z_LEN should be equal to EMBEDDING_SIZE
CHUNKS = 8
CBN_MLP_DIM = 512
RELEVANCE_FACTOR = 1
LEARNING_RATE = 2e-4
BETAS = (0, 0.999)
EPOCHS=6000
CHECKPOINT_INTERVAL=10
