# Project: seq_generator

## Project Structure

### models/
- **__init__.py**: Initialization file for the models module.
- **transformer.py**: Implementation of the Transformer model.
- **bigram.py**: Implementation of the Bigram model.
- **mlp.py**: Implementation of the Multi-Layer Perceptron (MLP) model.
- **rnn.py**: Implementation of the Recurrent Neural Network (RNN) model.
- **gru.py**: Implementation of the Gated Recurrent Unit (GRU) model.
- **bow.py**: Implementation of the Bag of Words (BoW) model.

### utils/
- **__init__.py**: Initialization file for the utils module.
- **data_loader.py**: Contains functions and classes for loading and processing data.
- **vocabulary.py**: Contains class for generating vocab and tokenization methods

### scripts/
- **train.py**: Script for training models.
- **sample.py**: Script for generating samples from trained models.
- **evaluation.py**: Functions for evaluating model performance.

### tests/
- **__init__.py**: Initialization file for the tests module.
- **test_models.py**: Unit tests for model implementations.
- **test_utils.py**: Unit tests for utility functions.
- **test_data_loader.py**: Unit tests for data loading functions.

### .gitignore
- Specifies files and directories to be ignored by Git.

### README.md
- Provides an overview of the project, including structure and usage instructions.

### requirements.txt
- Lists the dependencies required to run the project.

### setup.py
- Script for setting up the project, including installation of dependencies and package configuration.

## Testing Instructions

To test the data loader module, you can run the following script:
```
from utils.data_loader import get_infinite_data_loader
file_path = 'data/names.txt'
batch_size = 32
sentences_loader, vocab = get_infinite_data_loader(file_path, batch_size, transform=transform)
for batch in sentences_loader:
  print(batch.shape)
  break
```
