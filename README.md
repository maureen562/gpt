# GPT-style Language Model Project

This repository contains a small GPT-style character-level language model trained on *Pride and Prejudice* by Jane Austen.

The model follows the basic Transformer decoder structure used in GPT-style language models, including token embeddings, positional encoding, masked self-attention, multi-head attention, feed-forward layers, residual connections, and layer normalization.

## Dataset

The dataset used in this project is *Pride and Prejudice* from Project Gutenberg.

The text is downloaded automatically when the script is executed:

```python
DATA_URL = "https://www.gutenberg.org/ebooks/1342.txt.utf-8"
```

The model is trained at the character level, so each character in the text is treated as a token.

## Model Overview

The implemented model is a small educational GPT-style language model, not a full-scale GPT-2 model.

Main components:

* Character-level tokenization
* Sequence dataset for next-character prediction
* Masked self-attention
* Multi-head attention
* Feed-forward network
* Transformer blocks
* Text generation by sampling from predicted probabilities

## Training Settings

The main training settings are:

* Block size: 64
* Batch size: 64
* Embedding dimension: 128
* Number of attention heads: 4
* Number of Transformer blocks: 4
* Dropout: 0.1
* Learning rate: 3e-4
* Maximum steps per epoch: 300

The final model was trained using the 150-epoch experiment.

## Experiments

To compare the effect of training length, three experiments were saved: 50 epochs, 100 epochs, and 150 epochs.

| Experiment | Final Train Loss | Final Validation Loss | Best Validation Loss |
| ---------- | ---------------: | --------------------: | -------------------: |
| 50 epochs  |           1.2393 |                1.1753 |               1.1753 |
| 100 epochs |           1.1538 |                1.1298 |               1.1298 |
| 150 epochs |           1.1127 |                1.1277 |               1.1216 |

The 150-epoch experiment achieved the lowest validation loss. The best validation loss was 1.1216 at epoch 144, so the checkpoint from the best validation point was saved as the final model checkpoint.

## Generated Text

Generated text samples are included for each experiment.
They show how the model output changes as the number of training epochs increases.

## Files

```text
gpt2_prideandprejudice_epoch150.py
README.md

epoch_050/
  training_log_50.csv
  generated_text_50.txt

epoch_100/
  training_log_100.csv
  generated_text_100.txt

epoch_150/
  training_log_150.csv
  generated_text_150.txt
  gpt2_pride_checkpoint.pt
