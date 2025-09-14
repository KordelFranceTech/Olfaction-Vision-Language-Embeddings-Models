import torch
from PIL import Image

from src import constants
from src import base_model as bm
from src import graph_model as gm


if __name__ == "__main__":

    # Build example vision-olfaction sample with dummy data
    example_image = Image.new('RGB', (constants.IMG_DIM, constants.IMG_DIM))
    example_image.save(f"/tmp/image_example.jpg")
    example_olf_vec = torch.randn(constants.AROMA_VEC_LENGTH)

    # -------- Option A --------
    # Load the base models
    vision_lang_encoder, olf_encoder, graph_model = bm.load_model()
    # Get embeddings from base models
    ovl_embeddings_base = bm.run_inference(
        vision_lang_encoder=vision_lang_encoder,
        olf_encoder=olf_encoder,
        graph_model=graph_model,
        image=example_image,
        olf_vec=example_olf_vec
    )
    print(f"Olfaction-Vision-Language Embeddings from Base Model: {ovl_embeddings_base}")

    # -------- Option B --------
    # Load the graph attention models
    vision_lang_encoder, olf_encoder, graph_model = gm.load_model()
    # Get embeddings from graph attention models
    ovl_embeddings_graph = gm.run_inference(
        vision_lang_encoder=vision_lang_encoder,
        olf_encoder=olf_encoder,
        graph_model=graph_model,
        image=example_image,
        olf_vec=example_olf_vec
    )
    print(f"Olfaction-Vision-Language Embeddings from Graph Attention Model: {ovl_embeddings_graph}")

