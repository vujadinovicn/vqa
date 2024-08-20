import torch
from torch import nn


class VQAStackModel(nn.Module):
    def __init__(self, num_classes=1001, train=True):
        super(VQAStackModel, self).__init__()

        self.img_proj = nn.Linear(768, 1024)
        self.txt_proj = nn.Linear(768, 1024)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1000),

            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)
        )

        self.train = train


    def forward(self, text_features, image_features):
        assert text_features.shape == image_features.shape, "Feature dimensions must match for point-wise multiplication."

        image_features = self.img_proj(image_features)
        text_features = self.txt_proj(text_features)

        fused_features = torch.cat((text_features, image_features), dim=2) if self.train else torch.cat((text_features, image_features), dim=1)
        logits = self.mlp(fused_features)

        return logits