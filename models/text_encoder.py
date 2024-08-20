from torch import nn
import torch
from transformers import BertTokenizer, BertModel

class TextEncoderBERT(nn.Module):
  def __init__(self, pretrained_model_name='bert-base-uncased'):
    super(TextEncoderBERT, self).__init__()
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model = BertModel.from_pretrained(pretrained_model_name).to(self.device)
    self.model.eval()
    self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

  def forward(self, question):
    inputs = self.tokenizer(question, return_tensors='pt', padding=True, truncation=True).to(self.device)
    outputs = self.model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    cls_output = last_hidden_state[:, 0, :]
    return cls_output