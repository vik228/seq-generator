import torch
class Trainer:

    def __init__(self, model, train_data_loader, learning_rate) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.train_data_loader = train_data_loader
    
    def train(self):
        optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
