import torch
from torchvision import transforms
from transformers import CLIPModel, SiglipModel

from src import constants


# INFERENCE
def run_inference(vision_lang_encoder, olf_encoder, graph_model, image, olf_vec):
    vision_lang_encoder.eval()
    olf_encoder.eval()
    graph_model.eval()

    transform = transforms.Compose([
        transforms.Resize((constants.IMG_DIM, constants.IMG_DIM)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(constants.DEVICE)
    olf_tensor = torch.tensor(olf_vec, dtype=torch.float32).unsqueeze(0).to(constants.DEVICE)

    with torch.no_grad():
        vision_embed = vision_lang_encoder.get_image_features(pixel_values=image_tensor)
        olf_embed = olf_encoder(olf_tensor)

        nodes = torch.cat([vision_embed, olf_embed], dim=0)
        edge_index = torch.cartesian_prod(torch.arange(nodes.size(0)), torch.arange(nodes.size(0))).T.to(constants.DEVICE)
        embeds_final = graph_model(nodes, edge_index)

    return embeds_final


def load_model():
    # Use CLIP as default baseline
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(constants.DEVICE)
    clip_model.eval()
    """
    Or, you can also use SigLIP:
        SiglipModel.from_pretrained(
            "google/siglip-so400m-patch14-384",
            attn_implementation="flash_attention_2",
            dtype=torch.float16,
            device_map=constants.DEVICE,
        )
    """
    olf_encoder = torch.load(constants.ENCODER_SMALL_GRAPH_PATH, weights_only=False).to(constants.DEVICE)
    olf_encoder.eval()
    graph_model = torch.load(constants.OVLE_SMALL_GRAPH_PATH, weights_only=False).to(constants.DEVICE)
    graph_model.eval()

    return clip_model, olf_encoder, graph_model

