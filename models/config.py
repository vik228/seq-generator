from dataclasses import dataclass
from utils.data_loader import SentencesDataset

@dataclass
class Config:
    input_seq_len = None # Input Sequence Length
    output_seq_len = None # Output Sequence Length
    vocab_size = None # Number of items in vocabulary
    padding_values = SentencesDataset.PADDING_VALUE

