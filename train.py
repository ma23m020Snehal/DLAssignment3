# train.py

import os
import torch
import random
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path

# Set global seed
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Utility Functions
# ----------------------------
def load_dakshina_pairs(path):
    df = pd.read_csv(path, sep="\t", names=["native", "latin", "count"], dtype=str)
    df = df.dropna(subset=["native", "latin"])
    df["native"] = df["native"].str.strip()
    df["latin"] = df["latin"].str.strip()
    df = df[(df["native"] != "") & (df["latin"] != "")]
    return list(zip(df["latin"], df["native"]))

def build_vocab(pairs, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
    input_chars = sorted(set("".join(src for src, _ in pairs)))
    target_chars = sorted(set("".join(trg for _, trg in pairs)))
    input_vocab = specials + input_chars
    target_vocab = specials + target_chars
    return input_vocab, target_vocab

# ----------------------------
# Dataset Class
# ----------------------------
class TransliterationDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, in2idx, out2idx, max_in, max_out):
        self.pairs, self.in2idx, self.out2idx = pairs, in2idx, out2idx
        self.max_in, self.max_out = max_in, max_out
        self.pad_i, self.pad_o = in2idx["<pad>"], out2idx["<pad>"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        latin, deva = self.pairs[idx]
        src = [self.in2idx["<sos>"]] + [self.in2idx.get(c, self.in2idx["<unk>"]) for c in latin] + [self.in2idx["<eos>"]]
        tgt = [self.out2idx["<sos>"]] + [self.out2idx.get(c, self.out2idx["<unk>"]) for c in deva] + [self.out2idx["<eos>"]]
        src = src[:self.max_in] + [self.pad_i] * (self.max_in - len(src))
        tgt = tgt[:self.max_out] + [self.pad_o] * (self.max_out - len(tgt))
        return torch.tensor(src), torch.tensor(tgt)

# ----------------------------
# Vanilla Seq2Seq Model
# ----------------------------
class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, config):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['hid_dim']
        self.bidirectional = config['bidir']
        rnn_cls = getattr(nn, config['cell_type'])
        self.src_emb = nn.Embedding(input_vocab_size, self.emb_dim)
        self.tgt_emb = nn.Embedding(target_vocab_size, self.emb_dim)
        self.encoder = rnn_cls(self.emb_dim, self.hid_dim, 1, batch_first=True, bidirectional=self.bidirectional)
        self.decoder = rnn_cls(self.emb_dim, self.hid_dim, 1, batch_first=True)
        self.fc_out = nn.Linear(self.hid_dim, target_vocab_size)
        if self.bidirectional:
            self.bridge = nn.Linear(self.hid_dim * 2, self.hid_dim)

    def forward(self, src, tgt, tf_ratio=1.0):
        enc_embed = self.src_emb(src)
        dec_embed = self.tgt_emb(tgt)
        enc_outs, hidden = self.encoder(enc_embed)
        if self.bidirectional:
            if isinstance(hidden, tuple):
                h, c = hidden
                h_cat = torch.cat([h[-2], h[-1]], dim=1)
                c_cat = torch.cat([c[-2], c[-1]], dim=1)
                h = self.bridge(h_cat).unsqueeze(0)
                c = self.bridge(c_cat).unsqueeze(0)
                hidden = (h, c)
            else:
                h = hidden
                h_cat = torch.cat([h[-2], h[-1]], dim=1)
                hidden = self.bridge(h_cat).unsqueeze(0)
        dec_outs, _ = self.decoder(dec_embed, hidden)
        return self.fc_out(dec_outs)

# ----------------------------
# Attention Mechanism
# ----------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, dec_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, dec_dim, bias=False)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, dec_h, enc_outs, mask):
        score = self.v(torch.tanh(self.W_enc(enc_outs) + self.W_dec(dec_h).unsqueeze(1))).squeeze(-1)
        score.masked_fill_(mask == 0, float('-inf'))
        attn = torch.softmax(score, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_outs).squeeze(1)
        return ctx, attn

# ----------------------------
# Seq2Seq with Attention
# ----------------------------
class Seq2SeqAttn(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, config):
        super().__init__()
        self.emb_dim = config['emb_dim']
        self.hid_dim = config['hid_dim']
        self.bidirectional = config['bidir']
        rnn_cls = getattr(nn, config['cell_type'])
        self.src_emb = nn.Embedding(input_vocab_size, self.emb_dim)
        self.tgt_emb = nn.Embedding(target_vocab_size, self.emb_dim)
        self.encoder = rnn_cls(self.emb_dim, self.hid_dim, 1, batch_first=True, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.bridge = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.attn = BahdanauAttention(self.hid_dim * (2 if self.bidirectional else 1), self.hid_dim)
        self.decoder = rnn_cls(self.emb_dim + self.hid_dim * (2 if self.bidirectional else 1), self.hid_dim, 1, batch_first=True)
        self.fc_out = nn.Linear(self.hid_dim, target_vocab_size)

    def _init_decoder_state(self, enc_h):
        if self.bidirectional:
            if isinstance(enc_h, tuple):
                h, c = enc_h
                h = torch.cat([h[-2], h[-1]], dim=1)
                c = torch.cat([c[-2], c[-1]], dim=1)
                h = self.bridge(h).unsqueeze(0)
                c = self.bridge(c).unsqueeze(0)
                return (h, c)
            else:
                h = enc_h
                h = torch.cat([h[-2], h[-1]], dim=1)
                return self.bridge(h).unsqueeze(0)
        return enc_h

    def forward(self, src, tgt, tf_ratio=1.0):
        B, S = src.size()
        _, T = tgt.size()
        mask = (src != 0)
        enc_outs, enc_h = self.encoder(self.src_emb(src))
        dec_h = self._init_decoder_state(enc_h)
        logits = torch.zeros(B, T, self.fc_out.out_features, device=src.device)
        inp = tgt[:, 0]
        for t in range(1, T):
            emb = self.tgt_emb(inp).unsqueeze(1)
            last_h = dec_h[0][-1] if isinstance(dec_h, tuple) else dec_h[-1]
            ctx, _ = self.attn(last_h, enc_outs, mask)
            dec_in = torch.cat([emb, ctx.unsqueeze(1)], dim=2)
            out, dec_h = self.decoder(dec_in, dec_h)
            pred = self.fc_out(out.squeeze(1))
            logits[:, t] = pred
            inp = tgt[:, t] if torch.rand(1).item() < tf_ratio else pred.argmax(1)
        return logits

# ----------------------------
# Main Training Workflow
# ----------------------------
def main(args):
    # Load data
    train_pairs = load_dakshina_pairs(args.train_path)
    dev_pairs = load_dakshina_pairs(args.dev_path)
    test_pairs = load_dakshina_pairs(args.test_path)

    # Build vocab and mappings
    src_vocab, tgt_vocab = build_vocab(train_pairs)
    s2i = {c: i for i, c in enumerate(src_vocab)}
    t2i = {c: i for i, c in enumerate(tgt_vocab)}
    i2t = {i: c for c, i in t2i.items()}

    # Special IDs
    PAD, SOS, EOS = t2i['<pad>'], t2i['<sos>'], t2i['<eos>']

    # Max lengths
    max_src = max(len(x[0]) for x in train_pairs) + 2
    max_tgt = max(len(x[1]) for x in train_pairs) + 2

    # DataLoaders
    train_ds = TransliterationDataset(train_pairs, s2i, t2i, max_src, max_tgt)
    dev_ds   = TransliterationDataset(dev_pairs,   s2i, t2i, max_src, max_tgt)
    test_ds  = TransliterationDataset(test_pairs,  s2i, t2i, max_src, max_tgt)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # Config dict
    config = {
        'emb_dim': args.embedding_dim,
        'hid_dim': args.hidden_size,
        'cell_type': args.cell_type,
        'bidir': args.bidirectional
    }

    # Model selection
    if args.attention:
        model = Seq2SeqAttn(len(src_vocab), len(tgt_vocab), config).to(DEVICE)
    else:
        model = Seq2Seq(len(src_vocab), len(tgt_vocab), config).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Evaluation function
    def evaluate(loader):
        model.eval()
        corr_chars = tot_chars = corr_words = tot_words = 0
        with torch.no_grad():
            for src, tgt in loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                logits = model(src, tgt if args.attention else tgt[:, :-1])
                preds = logits.argmax(-1)

                # character accuracy
                gold = tgt if args.attention else tgt[:, 1:]
                mask = gold != PAD
                corr_chars += (preds == gold).masked_select(mask).sum().item()
                tot_chars += mask.sum().item()

                # exact match
                for p_seq, g_seq in zip(preds, gold):
                    p = p_seq.tolist()
                    g = g_seq.tolist()
                    if EOS in p: p = p[:p.index(EOS)]
                    if EOS in g: g = g[:g.index(EOS)]
                    if ''.join(i2t[i] for i in p) == ''.join(i2t[i] for i in g):
                        corr_words += 1
                    tot_words += 1
        return corr_chars/tot_chars, corr_words/tot_words

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0
        for src, tgt in tqdm(train_loader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, tgt if args.attention else tgt[:, :-1])
            gold = tgt if args.attention else tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        val_char_acc, val_word_acc = evaluate(dev_loader)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Char Acc: {val_char_acc:.4f} | Val Word Acc: {val_word_acc:.4f}")

    # Save best model
    torch.save(model.state_dict(), "best_model.pt")
    print("Model saved to best_model.pt")

    # Evaluate on test set
    test_char_acc, test_word_acc = evaluate(test_loader)
    print(f"Test Char Acc: {test_char_acc:.4f} | Test Word Acc: {test_word_acc:.4f}")

# ----------------------------
# Argument Parser
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--dev_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--cell_type', type=str, default="LSTM")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--attention', action='store_true', help='Use attention-based model')
    args = parser.parse_args()
    main(args)
