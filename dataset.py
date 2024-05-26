# dataset.py

import os
import json
import torch
from torch.utils.data import Dataset


class CRDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'token_ids': torch.tensor(self.features[idx]["token_ids"], dtype=torch.long),
            'attention_masks': torch.tensor(self.features[idx]["attention_masks"], dtype=torch.long),
            'token_type_ids': torch.tensor(self.features[idx]["token_type_ids"], dtype=torch.long),
            'span1_ids': torch.tensor(self.features[idx]["span1_ids"], dtype=torch.long),
            'span2_ids': torch.tensor(self.features[idx]["span2_ids"], dtype=torch.long),
            'label': torch.tensor(self.features[idx]["label"], dtype=torch.long)
        }


def get_data(processor, file_path, mode, args):
    data_dir = args.data_dir
    examples = processor.get_examples(os.path.join(data_dir, file_path))
    features = processor.convert_examples_to_features(examples, args)
    return features
