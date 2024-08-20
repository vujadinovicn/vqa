import argparse
import torch
from PIL import Image
from utils.files import load_json_data, load_yaml_data
from models.vqa_stack_model import VQAStackModel
from models.text_encoder import TextEncoderBERT
from models.image_encoder import ImageEncoderDinoV2

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

def inference(image, question):
    config = load_yaml_data("config.yml")

    image_encoder = ImageEncoderDinoV2()
    text_encoder = TextEncoderBERT()
    model = VQAStackModel(num_classes=config.get("num_classes"), train=False)
    load_checkpoint(model, config.get("checkpoint_path"))

    txt_tens = text_encoder(question)
    image_tens = image_encoder(image)

    result = model(txt_tens, image_tens)
    lbl = torch.argmax(result, dim=1)

    labels = load_json_data(config.get("most_common_answers_file_path"))
    answer = list(labels.keys())[lbl]
    if (answer == "<unknown>"):
        answer = "Sorry, I do not know the answer to that question :("

    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a VQA model.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--question', type=str, required=True, help="Question related to the image.")

    args = parser.parse_args()
    answer = inference(Image.open(args.image), args.question)
    
    print(f"Answer: {answer}")