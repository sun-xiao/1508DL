from collections import Counter
from typing import List


class Vocab:
    def __init__(self, texts: List[str], max_size: int = 20000, min_freq: int = 2):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        counter = Counter()
        for text in texts:
            counter.update(text.split())

        self.itos = [self.pad_token, self.unk_token]

        for word, freq in counter.most_common():
            if freq < min_freq:
                break
            if len(self.itos) >= max_size:
                break
            self.itos.append(word)

        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

        self.pad_idx = self.stoi[self.pad_token]
        self.unk_idx = self.stoi[self.unk_token]

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        return [self.stoi.get(token, self.unk_idx) for token in tokens]