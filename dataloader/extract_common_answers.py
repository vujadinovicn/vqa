from utils.files import load_json_data, dump_json_data
from collections import Counter

if __name__ == "__main__":
    annotations_file_path = 'processed_train_annotations.json'
    output_file_path = 'most_common_train_answers.json'
    processed_train_annotations_data = load_json_data(annotations_file_path)

    train_answers = []
    for entry in processed_train_annotations_data:
        train_answers.append(entry['answer'])

    most_common_answers = Counter(train_answers).most_common(1000)
    answer_to_index = {answer: index for index, (answer, _) in enumerate(most_common_answers)}
    answer_to_index["<unknown>"] = 1000

    dump_json_data(answer_to_index, output_file_path)
