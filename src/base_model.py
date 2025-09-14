import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
from torch_geometric.nn import GATConv
from PIL import Image

from src.constants import AROMA_VEC_LENGTH, EMBED_DIM, IMG_DIM


# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dummy_olfaction_dim = len(AROMA_VEC_LENGTH)  # Number of olfactory features per input


# DATASET EXAMPLE
class OlfactionVisionDataset(Dataset):
    def __init__(self, image_paths, olfaction_vectors, labels):
        self.image_paths = image_paths
        self.olfaction_vectors = olfaction_vectors
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((IMG_DIM, IMG_DIM)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.transform(Image.open(img_path).convert('RGB'))
        olf_vec = self.olfaction_vectors[idx]
        label = self.labels[idx]
        return image, torch.tensor(olf_vec, dtype=torch.float32), label

# ENCODERS
class OlfactionEncoder(nn.Module):
    def __init__(self, input_dim=dummy_olfaction_dim, embed_dim=EMBED_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, AROMA_VEC_LENGTH),
            nn.ReLU(),
            nn.Linear(AROMA_VEC_LENGTH, embed_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Note that this graph is a GAT and the specialized layers may be problematic for export
# GRAPH MODULE
class GATGraphAssociator(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.gat1 = GATConv(embed_dim, embed_dim, heads=16, concat=True)
        self.gat2 = GATConv(embed_dim * 16, embed_dim, heads=1, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return self.classifier(x)


# INFERENCE
def run_inference(clip_model, olf_encoder, graph_model, image, olf_vec):
    clip_model.eval()
    olf_encoder.eval()
    graph_model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_DIM, IMG_DIM)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    olf_tensor = torch.tensor(olf_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        vision_embed = clip_model.get_image_features(pixel_values=image_tensor)
        olf_embed = olf_encoder(olf_tensor)

        nodes = torch.cat([vision_embed, olf_embed], dim=0)
        edge_index = torch.cartesian_prod(torch.arange(nodes.size(0)), torch.arange(nodes.size(0))).T.to(DEVICE)
        logits = graph_model(nodes, edge_index)
        prediction = torch.sigmoid(logits[0])

    return prediction.cpu().numpy()


if __name__ == "__main__":

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_model.train()  # Enable fine-tuning
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    olf_encoder = OlfactionEncoder().to(DEVICE)
    graph_model = GATGraphAssociator().to(DEVICE)
    example_image = Image.new('RGB', (IMG_DIM, IMG_DIM))
    example_image.save(f"/tmp/image_example.jpg")
    example_olf_vec = torch.randn(dummy_olfaction_dim)
    pred = run_inference(
        clip_model,
        olf_encoder,
        graph_model,
        example_image,
        example_olf_vec
    )
    print("Predicted label scores:", pred)

