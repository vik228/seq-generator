import torch
import torch.nn.functional as F

class NNBigram:

    def __init__(self):
        self.file_path = 'data/names.txt'
        self.names = None
        self.itos = {}
        self.stoi = {}
        self.g = torch.Generator().manual_seed(2147483647)
        self.load_and_prepare_data()
        self.num_epochs = 200
        self.W = torch.randn((27, 27), generator=self.g, requires_grad=True)
    
    def load_and_prepare_data(self):
        self.names = open(self.file_path).read().splitlines()
        all_chars = sorted(list(set(''.join(self.names))))
        self.itos = {(i+1):char for i, char in enumerate(all_chars)}
        self.stoi = {char:(i+1) for i, char in enumerate(all_chars)}
        self.itos[0] = '.'
        self.stoi['.'] = 0
    
    def forward(self):
        xs = []
        ys = []
        for word in self.names:
            word = ['.'] + list(word) + ['.']
            for ch1, ch2 in zip(word, word[1:]):
                idx1 = self.stoi[ch1]
                idx2 = self.stoi[ch2]
                xs.append(idx1)
                ys.append(idx2)
        xs = torch.tensor(xs)
        num = xs.nelement()
        ys = torch.tensor(ys)
        xenc = F.one_hot(xs, num_classes=27).float()
        for i in range(self.num_epochs):
            logits = xenc @ self.W
            counts = logits.exp()
            probs = counts/counts.sum(1, keepdims=True)
            loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(self.W**2).mean()
            print(loss.item())
            self.W.grad = None
            loss.backward()
            self.W.data += -50*self.W.grad
    
    def predict_next(self, n):
        ix = 0
        for i in range(n):
            out = []
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                logits = xenc @ self.W
                counts = logits.exp()
                probs = counts/counts.sum(1, keepdims=True)
                ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=self.g).item()
                out.append(self.itos[ix])
                if ix == 0:
                    break
            yield ''.join(out)



