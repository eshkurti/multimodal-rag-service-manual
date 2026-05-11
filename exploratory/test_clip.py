"""
Exploratory: CLIP cross-modal similarity analysis.

Tests whether CLIP embeddings can reliably match text queries to technical
service manual images (tables, schematics, natural car photos).

Conclusion: CLIP similarity scores for technical automotive content are
consistently below 0.30, making it unsuitable as the primary retrieval
mechanism for this domain. See the term paper for full analysis.

Requirements: pip install transformers torch
"""

import sys
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def compute_clip_similarity(image_path: str, query: str) -> float:
    """Compute cosine similarity between a text query and an image using CLIP."""
    image = Image.open(image_path).convert("RGB")

    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor(text=query, return_tensors="pt", padding=True)

    with torch.no_grad():
        image_emb = F.normalize(model.get_image_features(**image_inputs), p=2, dim=-1)
        text_emb = F.normalize(model.get_text_features(**text_inputs), p=2, dim=-1)

    return (image_emb @ text_emb.T).item()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_clip.py <image_path> <query>")
        sys.exit(1)

    image_path = sys.argv[1]
    query = " ".join(sys.argv[2:])
    similarity = compute_clip_similarity(image_path, query)
    print(f"CLIP cosine similarity: {similarity:.4f}")
