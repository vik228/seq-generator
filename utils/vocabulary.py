import torch
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:

    def __init__(self) -> None:
        self.token_to_idx = {}
        self.idx_to_token = []
        self.add_token("<UNK>")
        self.add_token("<SOS>")
        self.add_token("<EOS>")
    
    def add_token(self, token):
        if token not in self.token_to_idx:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)

    def token_to_index(self, token):
        return self.token_to_idx.get(token, self.token_to_idx.get("<UNK>"))

    def index_to_token(self, index):
        return self.index_to_token[index]
    
    def tokens_to_indices(self, tokens):
        return [self.token_to_idx[token] for token in tokens]
    
    def indices_to_tokens(self, indices):
        return [self.idx_to_token[idx] for idx in indices if idx >= 0]
    
    def build_vocabulary_from_dataset(self, dataset):
        for sentence in dataset:
            for char in sentence:
                self.add_token(char)
    
    def tokenize_and_convert_to_indices(self, sentence):
        tokens = list(sentence)
        indices = self.tokens_to_indices(tokens)
        return indices