import torch
import torch.nn as nn
import torch.nn.functional as F

class VQAMulModel(nn.Module):
    def __init__(self, num_classes, input_dim=768):
        super(VQAMulModel, self).__init__()

        self.img_proj = nn.Linear(input_dim, 1024)
        self.txt_proj = nn.Linear(input_dim, 1024)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1000),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)
        )

    def forward(self, text_features, image_features):
        assert text_features.shape == image_features.shape, "Feature dimensions must match for element-wise multiplication."

        image_features = self.img_proj(image_features)
        text_features = self.txt_proj(text_features)

        image_features = F.normalize(image_features, p=2, dim=1)
        fused_features = torch.mul(text_features, image_features)

        logits = self.mlp(fused_features)

        return logits
