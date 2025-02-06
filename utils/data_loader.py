import torch
import torch.utils
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.vocabulary import Vocabulary
from torch.nn.utils.rnn import pad_sequence

class SentencesDataset(Dataset):

    def __init__(self, dataset, vocab, tokenizer, transform=None, add_sos=None, add_eos=None, input_seq_len=None):
        self.dataset = dataset
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.add_sos = add_sos
        self.add_eos = add_eos
        self.transform = transform
        self.input_seq_len = input_seq_len
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        if self.transform:
            data = self.transform(data)
        tokens = self.tokenizer(data)
        token_indices = self.vocab.build_indices_from_tokens(tokens)
        if self.add_sos:
            token_indices = [self.vocab.get_idx_from_token('<SOS>')] + token_indices
        if self.add_eos:
            token_indices = token_indices + [self.vocab.get_idx_from_token('<EOS>')]
        if self.input_seq_len:
            tokens = tokens[:self.input_seq_len]
        sequence = torch.tensor(token_indices, dtype=torch.long)
        return sequence

def collate_fn(padding_value):
    def solve(batch):
        batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
        X = batch[:, :-1]
        Y = batch[:, 1:]
        return X, Y
    return solve

def get_infinite_data_loader(
    file_path,
    batch_size,
    tokenizer,
    add_sos=None,
    add_eos=None,
    transform=None,
    max_len=None,
    padding_value=-1,
    test_split=0.2
):
    vocab = Vocabulary()

    with open(file_path, 'r') as f:
        data = f.read()
    sentences = data.splitlines()
    sentences = [sentence.strip() for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence]
    
    all_tokens = []
    for sentence in sentences:
        if transform:
            sentence = transform(sentence)
        tokens = tokenizer(sentence)
        all_tokens.extend(tokens)
    vocab.build_indices_from_tokens(all_tokens)

    test_size = int(len(sentences) * test_split)
    train_size = len(sentences) - test_size

    train_sentences, test_sentences = torch.utils.data.random_split(
        sentences, [train_size, test_size]
    )

    train_dataset = SentencesDataset(
        train_sentences,
        vocab,
        tokenizer,
        transform=transform,
        add_eos=add_eos,
        add_sos=add_sos,
        input_seq_len=max_len
    )

    test_dataset = SentencesDataset(
        test_sentences,
        vocab,
        tokenizer,
        transform=transform,
        add_eos=add_eos,
        add_sos=add_sos,
        input_seq_len=max_len
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn(padding_value)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn(padding_value)
    )
    
    return train_loader, test_loader, vocab
        

