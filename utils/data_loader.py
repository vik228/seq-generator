import torch
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.vocabulary import Vocabulary
from torch.nn.utils.rnn import pad_sequence

class SentencesDataset(Dataset):

    def __init__(self, file_path, vocab=None, transform=None, max_len=None):
        self.sentences = self._load_data_from_file(file_path)
        self.transform = transform
        self.max_len = max_len
        if vocab:
            vocab.build_vocabulary_from_dataset(self.sentences)
        self.vocab = vocab
    
    def _load_data_from_file(self, file_path):
        with open(file_path, 'r') as file:
            sentences = file.readlines()
        sentences = [
            sentence.strip() for sentence in sentences
        ]
        return [
            sentence for sentence in sentences if sentence
        ]
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        if self.transform:
            sentence = self.transform(sentence)
        if self.vocab:
            sentence = self.vocab.tokenize_and_convert_to_indices(sentence)
            sentence = [self.vocab.token_to_index("<SOS>")] + sentence + [self.vocab.token_to_index("<EOS>")]
        if self.max_len:
            sentence = sentence[:self.max_len]
        return sentence

def collate_fn(batch):
    batch = [torch.tensor(item) for item in batch]
    batch = pad_sequence(batch, batch_first=True, padding_value=-1)
    return batch

def get_infinite_data_loader(file_path, batch_size, transform=None, max_len=None):
    vocab = Vocabulary()

    dataset = SentencesDataset(
        file_path, 
        vocab=vocab, 
        transform=transform, 
        max_len=max_len
    )
    sampler = RandomSampler(dataset, replacement=True)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn
    )
    return data_loader, vocab
        

