import torch
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:

    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.special_tokens = [
            '<UNK>',
            '<SOS>',
            '<EOS>'
        ]
    
    def build_indices_from_tokens(self, tokens):
        tokens = set(tokens)
        tokens = self.special_tokens + tokens
        self.idx_to_token = {i:token for i, token in enumerate(tokens)}
        self.token_to_idx = {token: i for i, token in enumerate(tokens)}
    
    def get_token_from_idx(self, idx):
        return self.idx_to_token.get(idx, self.token_to_idx['<UNK>'])

    def get_idx_from_token(self, token):
        return self.token_to_idx[token]