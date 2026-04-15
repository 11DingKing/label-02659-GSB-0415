import os

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "allenai/scibert_scivocab_cased")
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
