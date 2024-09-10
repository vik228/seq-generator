import torch

class BasicBigram:

    def __init__(self):
        self.file_path = 'data/names.txt'
        self.names = None
        self.chartoidx = {}
        self.idxtochar = {}
        self.bigram_freq = torch.zeros((27, 27), dtype=torch.int32)
        self.g = torch.Generator().manual_seed(2147483647)
        self.load_and_prepare_data()

    
    def load_and_prepare_data(self):
        self.names = open(self.file_path, 'r').read().splitlines()
        all_chars = sorted(list(set(''.join(self.names))))
        self.chartoidx['.'] = 0
        self.idxtochar[0] = '.'
        for i, ch in enumerate(all_chars):
            self.chartoidx[ch] = i+1
            self.idxtochar[i+1] = ch
        for name in self.names:
            chars = ['.'] + list(name) + ['.']
            for ch1, ch2 in zip(chars, chars[1:]):
                idx1 = self.chartoidx[ch1]
                idx2 = self.chartoidx[ch2]
                self.bigram_freq[idx1, idx2] += 1
        self.bigram_freq = (self.bigram_freq + 1).float()
        self.bigram_freq /= self.bigram_freq.sum(1, keepdims=True)
    
    def predict_next(self, n):
        for i in range(n):
            out = []
            idx = 0;
            while True:
                data = self.bigram_freq[idx]
                idx = torch.multinomial(data, num_samples=1, replacement=True, generator=self.g).item()
                if idx == 0:
                    break
                out.append(self.idxtochar[idx])
            yield ''.join(out)
    
    def find_prob(self, words):
        result = 0
        n = 0
        for name in words:
            name = ['.'] + list(name) + ['.']
            for ch1, ch2 in zip(name, name[1:]):
                ix1 = self.chartoidx[ch1]
                ix2 = self.chartoidx[ch2]
                p = torch.log(self.bigram_freq[ix1, ix2])
                result += p;
                n += 1
        result = -result
        return result/n
