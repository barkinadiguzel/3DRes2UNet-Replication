import torch

# Volume & patch config
PATCH_SIZE = (64, 128, 128)   
SPACING = (1.0, 1.0, 1.0)

# Model config
IN_CHANNELS = 1
NUM_CLASSES = 1

# Inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessing
HU_MIN = -1000
HU_MAX = 400
