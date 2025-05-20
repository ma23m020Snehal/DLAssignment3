# Hindi Transliteration using Seq2Seq Models (with and without Attention)

This repository implements a sequence-to-sequence transliteration model for converting Latin-script Hindi text to Devanagari. Two architectures are explored: a vanilla encoder-decoder model and an attention-enhanced model. Both models are trained and evaluated on the Hindi dataset from the Dakshina corpus by Google Research.

> **WandB Report Link**: [Click here](https://wandb.ai/snehalma23m020-iit-madras/dakshina-translit-hi/reports/MA23M020-Snehal-Assignment-3--VmlldzoxMjgyODE3Mg?accessToken=mwuqyuul7h9zxkq98ow5szffmeionw6d6ubch0x67fq4qwhj5yeo4jj0e8n0r1jw)

---

## Project Overview

Transliteration is the process of converting text from one script to another while preserving its pronunciation. This project uses PyTorch to build two types of models:
- **Vanilla Seq2Seq model**: Traditional encoder-decoder without attention.
- **Attention-based Seq2Seq model**: Augmented with attention to dynamically focus on relevant encoder states during decoding.

Models are trained with:
- Teacher Forcing
- Configurable RNN cell types: `RNN`, `GRU`, `LSTM`
- Optional bidirectionality and multi-layer RNNs
- Hyperparameter optimization using Weights & Biases sweeps

---
## Dataset

- Source: [Dakshina Dataset (Google Research)](https://github.com/google-research-datasets/dakshina)
- Files used:
  - `hi.translit.sampled.train.tsv`
  - `hi.translit.sampled.dev.tsv`
  - `hi.translit.sampled.test.tsv`

Each file contains: `<target_script_word> <latin_script_word>`

---

##  Model Architecture

###  Vanilla Seq2Seq
- **Encoder**:
  - Embedding → RNN (RNN/LSTM/GRU) → Final hidden state
- **Decoder**:
  - Embedding → RNN initialized with encoder final state → Linear projection to vocab

###  Attention-Based Seq2Seq
- **Encoder**: Same as above
- **Attention**: Bahdanau attention over encoder outputs
- **Decoder**:
  - Takes previous token and context vector as input
  - RNN → FC → Softmax

---

##  Hyperparameter Configuration

Best configuration (Vanilla model):
```python
embedding_dim  = 256
hidden_size    = 256
encoder_layers = 2
decoder_layers = 3
rnn_type       = 'LSTM'
dropout        = 0.3
beam_size      = 1
bidirectional  = False
```

Best configuration (Attention model):
```python
emb_dim        = 32
hid_dim        = 256
cell_type      = 'LSTM'
dropout        = 0.3
bidir          = True
batch_size     = 32
beam_size      = 5
epochs         = 10
lr             = 1e-3
tf_ratio       = 1.0
```

How to Run
Method 1: Use train.py
Train either model with Weights & Biases sweep:

????????????????????????????????????????????

Method 2: Use .ipynb on Kaggle/Colab

Open the provided notebook and modify paths based on environment:

```python
train_path = "/kaggle/input/snehal/hi.translit.sampled.train.tsv"
```

 Weights & Biases Sweep Configuration

```python
sweep_config = {
  'method': 'bayes',
  'metric': {'goal': 'maximize', 'name': 'val_word_acc'},
  'parameters': {
    'emb_dim':        {'values': [16, 32, 64, 128]},
    'hid_dim':        {'values': [64, 128, 256]},
    'dropout':        {'values': [0.1, 0.3, 0.5]},
    'cell_type':      {'values': ['lstm', 'gru', 'rnn']},
    'bidir':          {'values': [True, False]},
    'tf_ratio':       {'values': [0.5, 1.0]},
    'batch_size':     {'values': [32, 64]},
    'epochs':         {'values': [10, 15]},
    'lr':             {'values': [0.001, 0.0005]},
  }
}
```

## Results :

Model	                     Exact Match Accuracy (Test)

Vanilla Seq2Seq          	~0.3994 

Attention Seq2Seq	        ~0.4240



## Why Attention Helps ?

Dynamic Focus: Vanilla compresses input into a single vector; attention recalculates relevant context at each step.

Improved Rare Sequence Handling: Attention improves transliteration of rare or long sequences.

Better Performance: Fewer vowel/consonant misplacements (e.g., "ankganit" → "अंकगणित")


## Folder Structure

```
DL_Assignment_3/
├── predictions_attention/
│ └── predictions_attn.csv
├── predictions_vanilla/
│ └── predictions_vanilla.csv
├── attention_grid_heat_maps.png
├── q1to4-ass3-without-attention.ipynb
├── Q5 with-attention.ipynb
├── train.py
├── README.md
├── .gitignore
├── venv/ 

```
## Files info
q1to4-ass3-without-attention.ipynb - The main ipynb file containing the source code along with wandb sweeps for hyperparameter tuning without attention
 Q5 with-attention.ipynb - The main ipynb file containing the source code along with wandb sweeps for hyperparameter tuning with attention
train.py - Python file to train and run the code
prediction_vanilla.csv - Predicted data of the test set without attention
predictions_attn.csv - Predictions of the test set with Attention

## Output Files

predictions_vanilla.csv: Model predictions from the vanilla model


predictions_attention.csv: Predictions from attention-based model

Can be downloaded via githib
 Author
Snehal
MA23M020








