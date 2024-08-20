import torch
from torch import nn
from models.text_encoder import TextEncoderBERT
from models.image_encoder import ImageEncoderDinoV2
from collections import defaultdict
import os
from PIL import Image
from utils.files import load_json_data, dump_pickle_data

image_encoder = ImageEncoderDinoV2()
text_encoder = TextEncoderBERT()

def load_and_group_data_by_image(answers_file_path, questions_file_path):
    answers_data = load_json_data(answers_file_path)
    questions_data = load_json_data(questions_file_path)

    answers_by_image = defaultdict(list)
    for answer in answers_data:
        answers_by_image[answer['image_id']].append(answer)

    questions_by_image = defaultdict(list)
    for question in questions_data["questions"]:
        questions_by_image[question['image_id']].append(question)

    return answers_by_image, questions_by_image

def process_questions(text_encoder, answers_by_image, questions_by_image, image_id):
    combined_qas = []
    for answer in answers_by_image[image_id]:
        question_id = answer['question_id']
        question_text = next(
            (q['question'] for q in questions_by_image[image_id] if q['question_id'] == question_id), 
            None
        )
        with torch.no_grad():
            encoded_question = text_encoder(question_text).cpu()
        combined_qas.append({
            'question': encoded_question,
            'answer': answer['answer']
        })
    return combined_qas

def process_image(image_encoder, image_id):
    image_path = f"val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg"
    with torch.no_grad():
        image = Image.open(image_path)
        return image_encoder(image).cpu()
    

if __name__ == "__main__":
    image_encoder = ImageEncoderDinoV2()
    text_encoder = TextEncoderBERT()

    # change every 'val' to 'train' and every 'validation' to 'training'
    answers_file_path = 'data/processed_annotations.json'
    questions_file_path = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
    images_directory = 'data/val2014'
    output_directory = 'data/validation_set/'
    os.makedirs(output_directory, exist_ok=True)

    answers_by_image, questions_by_image = load_and_group_data_by_image(answers_file_path, questions_file_path)
    text_encoder.eval()

    for i, image_id in enumerate(answers_by_image, start=1):
        try:
            combined_qas = process_questions(text_encoder, answers_by_image, questions_by_image, image_id)
            image_output = process_image(image_encoder, image_id)

            output_data = {
                'image': image_output,
                'qas': combined_qas
            }

            dump_pickle_data(output_data, os.path.join(output_directory, f'{image_id}.pkl'))

            if i % 500 == 0:
                print(f"Finished {i} images")

        except Exception as e:
            print(f"Error: {str(e)} with image: {image_id}")