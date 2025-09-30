from transformers import AutoTokenizer, CLIPModel, AutoModel
from einops import rearrange
import torch
import pdb


def get_text_model(task_name, language_emb_model):
    if language_emb_model == "clip":
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    elif language_emb_model == "siglipv2":
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained("google/siglip2-so400m-patch16-256")
            text_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-256") # google/siglip2-base-patch32-256
    else:
        tokenizer = None
        text_model = None

    if "libero_10" in task_name:
        max_length = 30
    elif "umi" in task_name:
        max_length = 30
    else:
        max_length = 30

    return text_model, tokenizer, max_length


def extract_text_features(text_model, text_tokens, language_emb_model):
    with torch.no_grad():
        if language_emb_model in ["clip", "siglipv2"]:
            text_latents = text_model.get_text_features(**text_tokens)
        else:
            pdb.set_trace()

    return text_latents

def extract_image_features(image_model, images):
    assert images.dim() == 5 and images.size(1) == 3, f"Input images must be 4-dimensional with the third dimension equal to 3. Got shape {images.size()}"
    B, C, T, H, W = images.size()
    images = rearrange(images, "B C T H W -> (B T) C H W")  # Reshape to (B*T, C, H, W)
    # Check the value range of the tensor
    min_val, max_val = images.min().item(), images.max().item()
    with torch.no_grad():
        image_latents = image_model.get_image_features(images)

    # Reshape back to (B, T, C) if needed
    image_latents = image_latents.view(B, T, -1)
    return image_latents
