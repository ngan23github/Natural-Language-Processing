#!/usr/bin/env python
# coding: utf-8

import os
import random
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ------------------- Config -------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EMBED_DIM = 512
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
N_EPOCHS = 20
CLIP = 1.0
TEACHER_FORCING_RATIO = 0.5
MAX_LEN_DECODING = 50
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))

# ------------------- Tokenization -------------------
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# ------------------- Vocab -------------------
def yield_tokens(file_path, tokenizer):
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            yield tokenizer(line.strip().lower())

def build_vocab(file_path, tokenizer, max_tokens=10000):
    vocab = build_vocab_from_iterator(
        yield_tokens(file_path, tokenizer),
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        max_tokens=max_tokens
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

# Adjust paths
EN_TRAIN = './data/train.en'
FR_TRAIN = './data/train.fr'
EN_VAL = './data/val.en'
FR_VAL = './data/val.fr'

en_vocab = build_vocab(EN_TRAIN, en_tokenizer)
fr_vocab = build_vocab(FR_TRAIN, fr_tokenizer)

PAD_IDX_EN = en_vocab['<pad>']
PAD_IDX_FR = fr_vocab['<pad>']

# ------------------- Encode sentence -------------------
def encode_sentence(sentence, tokenizer, vocab):
    tokens = tokenizer(sentence.strip().lower())
    ids = [vocab['<sos>']] + [vocab[token] for token in tokens] + [vocab['<eos>']]
    return torch.tensor(ids, dtype=torch.long)

# ------------------- Dataset & Dataloader -------------------
class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, src_tokenizer, trg_tokenizer, src_vocab, trg_vocab):
        with open(src_file, encoding='utf-8') as f:
            self.src_lines = [l.strip() for l in f.readlines()]
        with open(trg_file, encoding='utf-8') as f:
            self.trg_lines = [l.strip() for l in f.readlines()]
        assert len(self.src_lines) == len(self.trg_lines)
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = encode_sentence(self.src_lines[idx], self.src_tokenizer, self.src_vocab)
        trg = encode_sentence(self.trg_lines[idx], self.trg_tokenizer, self.trg_vocab)
        return src, trg

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    trg_lengths = torch.tensor([len(t) for t in trg_batch], dtype=torch.long)
    sorted_idx = torch.argsort(src_lengths, descending=True)
    src_batch = [src_batch[i] for i in sorted_idx]
    trg_batch = [trg_batch[i] for i in sorted_idx]
    src_lengths = src_lengths[sorted_idx]
    trg_lengths = trg_lengths[sorted_idx]
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX_EN)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX_FR)
    return src_padded, trg_padded, src_lengths, trg_lengths

def get_dataloader(src_file, trg_file, batch_size=BATCH_SIZE, shuffle=True):
    dataset = TranslationDataset(src_file, trg_file, en_tokenizer, fr_tokenizer, en_vocab, fr_vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# ------------------- Model -------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed_emb = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.lstm(packed_emb)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, padding_value=0.0)
        return outputs, hidden, cell

class LuongAttention(nn.Module):
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim*2, output_dim)
        self.attention = LuongAttention()

    def forward(self, input_token, hidden, cell, encoder_outputs, src_mask=None):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        dec_hidden = hidden[-1]
        context, attn_weights = self.attention(dec_hidden, encoder_outputs, mask=src_mask)
        concat = torch.cat([output.squeeze(1), context], dim=1)
        prediction = self.fc_out(concat)
        return prediction, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, src_lengths, trg=None, teacher_forcing=True):
        batch_size = src.size(0)
        trg_len = trg.size(1) if trg is not None else MAX_LEN_DECODING
        output_dim = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        src_mask = (src != PAD_IDX_EN).to(self.device)
        input_token = trg[:,0] if trg is not None else torch.tensor([fr_vocab['<sos>']]*batch_size, device=self.device)
        for t in range(1, trg_len):
            teacher_force_flag = (random.random() < self.teacher_forcing_ratio) if (trg is not None and teacher_forcing) else False
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, src_mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = trg[:,t] if teacher_force_flag and trg is not None else top1
        return outputs

# ------------------- Training & Evaluation -------------------
def train_one_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for src, trg, src_lengths, trg_lengths in dataloader:
        src, trg, src_lengths = src.to(device), trg.to(device), src_lengths.to(device)
        optimizer.zero_grad()
        output = model(src, src_lengths, trg, teacher_forcing=True)
        output_dim = output.shape[-1]
        output_flat = output[:,1:,:].reshape(-1, output_dim)
        trg_flat = trg[:,1:].reshape(-1)
        loss = criterion(output_flat, trg_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    if dataloader is None:
        return float('inf')
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg, src_lengths, trg_lengths in dataloader:
            src, trg, src_lengths = src.to(device), trg.to(device), src_lengths.to(device)
            output = model(src, src_lengths, trg, teacher_forcing=False)
            output_dim = output.shape[-1]
            output_flat = output[:,1:,:].reshape(-1, output_dim)
            trg_flat = trg[:,1:].reshape(-1)
            loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX_FR)(output_flat, trg_flat)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# ------------------- Beam Search -------------------
def beam_translate(sentence: str, model, device, max_len=MAX_LEN_DECODING, beam_width=3):
    model.eval()
    src_tensor = encode_sentence(sentence, en_tokenizer, en_vocab).unsqueeze(0).to(device)
    src_lengths = torch.tensor([src_tensor.size(1)], device=device)
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lengths)
    sos_idx, eos_idx = fr_vocab['<sos>'], fr_vocab['<eos>']
    src_mask = (src_tensor != PAD_IDX_EN).to(device)
    beam = [( [sos_idx], hidden, cell, 0.0 )]
    completed = []

    for _ in range(max_len):
        new_beam = []
        for tokens, h, c, score in beam:
            if tokens[-1] == eos_idx:
                completed.append((tokens, score))
                continue
            input_token = torch.tensor([tokens[-1]], device=device)
            with torch.no_grad():
                pred, h_new, c_new, _ = model.decoder(input_token, h, c, encoder_outputs, src_mask)
                log_probs = F.log_softmax(pred, dim=1)
            topk_logp, topk_idx = torch.topk(log_probs, beam_width, dim=1)
            for i in range(beam_width):
                new_beam.append((tokens+[topk_idx[0,i].item()], h_new, c_new, score+topk_logp[0,i].item()))
        beam = sorted(new_beam, key=lambda x: x[3], reverse=True)[:beam_width]
        if not beam:
            break
    completed.extend(beam)
    best_tokens, _ = max(completed, key=lambda x: x[3]/len(x[0]))
    output_ids = best_tokens[1:]
    if eos_idx in output_ids:
        output_ids = output_ids[:output_ids.index(eos_idx)]
    words = [fr_vocab.get_itos()[i] for i in output_ids]
    return " ".join(words)

def compute_bleu(model, src_file, trg_file, device, max_samples=None, beam_width=3):
    smoothie = SmoothingFunction().method4
    scores = []
    with open(src_file, encoding='utf-8') as fsrc, open(trg_file, encoding='utf-8') as ftrg:
        for i, (sline, tline) in enumerate(zip(fsrc, ftrg)):
            if max_samples and i >= max_samples:
                break
            pred = beam_translate(sline.strip(), model, device, beam_width=beam_width)
            reference = [tline.strip().split()]
            hypothesis = pred.split()
            score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
            scores.append(score)
    return sum(scores)/len(scores) if scores else 0.0

# ------------------- Training -------------------
def run_training():
    enc = Encoder(len(en_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_EN)
    dec = Decoder(len(fr_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_FR)
    model = Seq2Seq(enc, dec, DEVICE, TEACHER_FORCING_RATIO).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_FR)
    train_loader = get_dataloader(EN_TRAIN, FR_TRAIN, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(EN_VAL, FR_VAL, batch_size=BATCH_SIZE, shuffle=False) if os.path.exists(EN_VAL) else None

    best_valid_loss = float('inf')
    patience = 3
    wait = 0

    for epoch in range(1, N_EPOCHS+1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CLIP, DEVICE)
        valid_loss = evaluate(model, val_loader, criterion, DEVICE) if val_loader else train_loss
        end = time.time()
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Time: {end-start:.1f}s")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            wait = 0
            print("  Saved new best model.")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
    return model

# ------------------- Main -------------------
if __name__ == "__main__":
    model = run_training()
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=DEVICE)
        enc = Encoder(len(en_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_EN)
        dec = Decoder(len(fr_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_FR)
        model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
        model.load_state_dict(data['model_state_dict'])
        print("Loaded best checkpoint.")

    TEST_FILE = './data/test_2016_flickr.en'
    with open(TEST_FILE, encoding='utf-8') as f:
        for i in range(5):
            s = f.readline().strip()
            print("EN:", s)
            print("FR_pred (beam):", beam_translate(s, model, DEVICE, beam_width=3))
            print("-"*30)

    if os.path.exists(EN_VAL) and os.path.exists(FR_VAL):
        bleu = compute_bleu(model, EN_VAL, FR_VAL, DEVICE, max_samples=200, beam_width=3)
        print("BLEU on val (avg, beam):", bleu)