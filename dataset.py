import os
import numpy as np
import torch
from torch.utils.data import Dataset

class GenreDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.genres = sorted(os.listdir(root_dir))
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

        for genre in self.genres:
            gpath = os.path.join(root_dir, genre)
            for f in os.listdir(gpath):
                self.samples.append((os.path.join(gpath, f), self.genre_to_idx[genre]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = np.load(path)
        mel = np.expand_dims(mel, axis=0)  # shape: (1, 128, time)
        mel = torch.tensor(mel, dtype=torch.float32)

        return mel, label
