import torch.nn as nn
import torch.nn.functional as F

class VQAModel(nn.Module):
    def __init__(self, num_classes, input_dim=768, num_heads=8):
        super(VQAModel, self).__init__()

        self.img_proj = nn.Linear(input_dim, 1024)
        self.txt_proj = nn.Linear(input_dim, 1024)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=num_heads)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1000),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)
        )

    def forward(self, text_features, image_features):
        assert text_features.shape == image_features.shape, "Feature dimensions must match for attention."

        image_features = self.img_proj(image_features)
        text_features = self.txt_proj(text_features)

        attn_output, _ = self.multihead_attn(text_features, image_features, image_features)
        fused_features = text_features + attn_output

        logits = self.mlp(fused_features)

        return logits
