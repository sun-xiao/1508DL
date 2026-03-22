import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=40):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def _pad_or_truncate(self, ids):
        if len(ids) < self.max_len:
            ids = ids + [self.vocab.pad_idx] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        ids = self.vocab.encode(text)
        ids = self._pad_or_truncate(ids)

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )