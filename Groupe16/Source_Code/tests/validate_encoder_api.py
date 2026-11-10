"""Offline check of the installed Transformers feature API, using random tiny models.

No pretrained weights, COCO data, or scientific embeddings are produced.
"""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, Dinov2Config, Dinov2Model

from src.embeddings.encode import Encoder


if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = CLIPConfig(
        text_config={"vocab_size": 64, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 16,
                     "bos_token_id": 0, "eos_token_id": 2, "pad_token_id": 1},
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "image_size": 16, "patch_size": 4},
        projection_dim=8,
    )
    model = CLIPModel(cfg).eval()
    adapter = Encoder.__new__(Encoder)
    adapter.torch, adapter.model, adapter.device, adapter.image_only = torch, model, "cpu", False
    adapter.processor = CLIPImageProcessor(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16})
    image = Image.new("RGB", (20, 16), (120, 80, 30))
    vectors = adapter.images([image])
    assert vectors.shape == (1, 8)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    with torch.inference_mode():
        text = adapter.features(model.get_text_features(input_ids=torch.tensor([[0, 5, 2]])))
    assert text.shape == (1, 8)
    # Same adapter branch for DINOv2: CLS image feature; no cross-modal comparison.
    adapter.model = Dinov2Model(Dinov2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                           image_size=16, patch_size=4)).eval()
    adapter.image_only = True
    assert adapter.images([image]).shape == (1, 16)
    print("Random tiny CLIP projected image/text and DINOv2 CLS API checks passed (no pretrained inference).")
