import pickle
import json
import yaml

def load_pickle_data(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_pickle_data(data, output_file_path):
    with open(output_file_path, 'wb') as f:
        pickle.dump(data, f)

def load_json_data(json_file_path):
    with open(json_file_path) as f:
        data = json.load(f)
    return data

def dump_json_data(data, output_file_path):
    with open(output_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_yaml_data(config_path):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return data
