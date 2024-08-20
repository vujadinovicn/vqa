import glob
import os
import torch
from torch.utils.data import Dataset
from utils.files import load_pickle_data

torch.manual_seed(0)

class VQADataset(Dataset):
    def __init__(self, folder="data/training_set", transform=None):
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        self.data = self._load_data()

    def _load_data(self):
        data_list = []
        for pickle_file in self.files:
            data = load_pickle_data(pickle_file)
            image = data['image']
            qas = data['qas']

            if self.transform:
                image = self.transform(image)

            for qa in qas:
                question = qa['question']
                answer = qa['answer']
                data_list.append([image, question, answer])

        return data_list

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)