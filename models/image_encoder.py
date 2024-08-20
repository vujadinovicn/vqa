import torch
from torch import nn
from torchvision import transforms

class ImageEncoderDinoV2(nn.Module):
    def __init__(self, pretrained_model_name="dinov2_vitb14"):
      super(ImageEncoderDinoV2, self).__init__()
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.model = torch.hub.load('facebookresearch/dinov2', pretrained_model_name).to(self.device)
      self.model.eval()

    def forward(self, image):
      image = self.preprocess_image(image).to(self.device)
      features = self.model(image)
      return features

    def preprocess_image(self, image):
      preprocess = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      image = preprocess(image).unsqueeze(0)
      return image
