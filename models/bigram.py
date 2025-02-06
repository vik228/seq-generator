
import torch
import torch.nn as nn
from models.config import Config
from torch.nn import Functional as F

class Bigram(nn.Module):

    def __init__(self, config: Config) -> None:
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros(n, n))
        self.config = config
    
    def forward(self, idx, targets=None):
        logits = self.logits[idx]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(
                    -1,
                    logits.size(-1) 
                ),
                targets.view(-1),
                ignore_index=self.config.padding_values
            )
        return logits, loss

