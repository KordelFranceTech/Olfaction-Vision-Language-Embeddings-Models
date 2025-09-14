import torch


# Number of features in each aroma vector
AROMA_VEC_LENGTH: int = 138
# All images are normalized to this
IMG_DIM: int = 224
# Each model was trained with these hyperparams
BATCH_SIZE: int = 16
EMBED_DIM: int = 512

# CPU or GPU?
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to models
OVLE_SMALL_BASE_PATH: str = "./model/ovle-small/base/gnn.pth"
ENCODER_SMALL_BASE_PATH: str = "./model/ovle-small/base/olf_encoder.pth"
OVLE_LARGE_BASE_PATH: str = "./model/ovle-large/base/gnn.pth"
ENCODER_LARGE_BASE_PATH: str = "./model/ovle-large/base/olf_encoder.pth"
OVLE_SMALL_GRAPH_PATH: str = "./model/ovle-small/graph/gat_gnn.pth"
ENCODER_SMALL_GRAPH_PATH: str = "./model/ovle-small/graph/gat_olf_encoder.pth"
OVLE_LARGE_GRAPH_PATH: str = "./model/ovle-large/graph/gat_gnn.pth"
ENCODER_LARGE_GRAPH_PATH: str = "./model/ovle-large/graph/gat_olf_encoder.pth"
