from utils.files import load_json_data, dump_json_data
from collections import Counter
import numpy as np
import Levenshtein as lev

def get_most_common_answer(counter):
    top_elements = counter.most_common(1)
    if len(top_elements) == 1:
        return top_elements[0][0]
    return None

def get_minimum_distance_answer(top_elements):
    current_min = np.inf
    current_answer = None
    
    for answer, _ in top_elements:
        total_distance = sum(lev.distance(answer, answer2) for answer2, _ in top_elements if answer != answer2)
        if total_distance < current_min:
            current_min = total_distance
            current_answer = answer
    
    return current_answer

def build_answer_vocab(entry):
    counter = Counter(answer_map['answer'] for answer_map in entry['answers'])
    
    chosen_answer = get_most_common_answer(counter)
    if not chosen_answer:
        top_elements = counter.most_common()
        chosen_answer = get_minimum_distance_answer(top_elements)
    
    entry['chosen_answer'] = chosen_answer
    return entry

def process_annotation(annotation):
    processed_annotation = build_answer_vocab(annotation)
    return {
        'question_type': processed_annotation['question_type'],
        'answer': processed_annotation['chosen_answer'],
        'image_id': processed_annotation['image_id'],
        'question_id': processed_annotation['question_id']
    }

if __name__ == "__main__":
    annotations_data = load_json_data('v2_mscoco_train2014_annotations.json')
    processed_annotations = [process_annotation(annotation) for annotation in annotations_data['annotations']]
    dump_json_data(processed_annotations, 'processed_train_annotations.json')

