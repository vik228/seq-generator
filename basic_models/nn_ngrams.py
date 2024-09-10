import torch
import torch.nn.functional as F


class NNNGram:

    def __init__(self, ngram) -> None:
        self.ngram = ngram
        self.file_path = 'data/names.txt'
        self.stoi = {}
        self.itos = {}
        self.g = torch.Generator().manual_seed(2147483647)
        self.names = None
    
    def load_and_prepare_data(self):
        self.names = open(self.file_path).read().splitlines()
        all_chars = sorted(list(set(''.join(self.names))))
        self.stoi = {ch:i+1 for i, ch in enumerate(all_chars)}
        self.itos = {i+1:ch for i, ch in enumerate(all_chars)}


