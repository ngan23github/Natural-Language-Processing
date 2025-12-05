#!/usr/bin/env python
# coding: utf-8

"""
Improved Seq2Seq with:
- Bidirectional encoder
- Luong-general attention
- Teacher forcing decay
- Label smoothing
- Beam search with length penalty
- Larger vocabulary
"""

import os
import random
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt

# --------------------
# Config (tweak here)
# --------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EMBED_DIM = 512
HIDDEN_DIM = 512        # decoder hidden size (encoder will be bidirectional => half per direction)
NUM_LAYERS = 2
DROPOUT = 0.3
N_EPOCHS = 20
CLIP = 1.0
TEACHER_FORCING_RATIO = 0.5
MAX_LEN_DECODING = 50
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Larger vocab to reduce <unk>
VOCAB_MAX_TOKENS = 30000

# files
EN_TRAIN = './data/train.en'
FR_TRAIN = './data/train.fr'
EN_VAL = './data/val.en'
FR_VAL = './data/val.fr'
TEST_FILE = './data/test_2016_flickr.en'

# reproducibility
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))


# --------------------
# Tokenizers & Vocab
# --------------------
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

def yield_tokens(file_path, tokenizer):
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            yield tokenizer(line.strip().lower())

def build_vocab(file_path, tokenizer, max_tokens=VOCAB_MAX_TOKENS):
    vocab = build_vocab_from_iterator(
        yield_tokens(file_path, tokenizer),
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        max_tokens=max_tokens
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

# Build (or reuse) vocabs
print("Building vocabularies (this can take a while)...")
en_vocab = build_vocab(EN_TRAIN, en_tokenizer, max_tokens=VOCAB_MAX_TOKENS)
fr_vocab = build_vocab(FR_TRAIN, fr_tokenizer, max_tokens=VOCAB_MAX_TOKENS)
PAD_IDX_EN = en_vocab['<pad>']
PAD_IDX_FR = fr_vocab['<pad>']

def encode_sentence(sentence: str, tokenizer, vocab):
    tokens = tokenizer(sentence.strip().lower())
    ids = [vocab['<sos>']] + [vocab[token] for token in tokens] + [vocab['<eos>']]
    return torch.tensor(ids, dtype=torch.long)


# --------------------
# Dataset & Dataloader
# --------------------
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


# --------------------
# Models
# --------------------
class Encoder(nn.Module):
    """
    Bidirectional LSTM encoder. We set hidden size per direction = HIDDEN_DIM // 2
    so that concatenated output size == HIDDEN_DIM
    """
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
        assert hidden_dim % 2 == 0, "hidden_dim must be even for bidirectional concat"
        hidden_per_dir = hidden_dim // 2
        self.lstm = nn.LSTM(embed_dim, hidden_per_dir, num_layers=num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, src, src_lengths):
        # src: [batch, src_len]
        embedded = self.embedding(src)  # [batch, src_len, embed_dim]
        packed_emb = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.lstm(packed_emb)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, padding_value=0.0)  # [batch, src_len, hidden_dim]
        # hidden/cell: [num_layers*2, batch, hidden_per_dir]
        return outputs, hidden, cell


class LuongGeneralAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        decoder_hidden: [batch, hidden_dim]
        encoder_outputs: [batch, src_len, hidden_dim]
        mask: [batch, src_len] (True for valid tokens)
        returns: context [batch, hidden_dim], attn_weights [batch, src_len]
        """
        # transform decoder hidden
        dec_trans = self.W(decoder_hidden)  # [batch, hidden_dim]
        # compute scores by dot product with encoder outputs
        scores = torch.bmm(encoder_outputs, dec_trans.unsqueeze(2)).squeeze(2)  # [batch, src_len]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=1)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, hidden_dim]
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        # fc_out: combine decoder output and context (concat) -> 2*hidden_dim
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.attention = LuongGeneralAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, src_mask=None):
        # input_token: [batch] (idx)
        input_token = input_token.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch,1,embed_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # output: [batch,1,hidden_dim]
        dec_hidden = hidden[-1]  # [batch, hidden_dim]
        context, attn_weights = self.attention(dec_hidden, encoder_outputs, mask=src_mask)
        concat = torch.cat([output.squeeze(1), context], dim=1)  # [batch, hidden_dim*2]
        prediction = self.fc_out(concat)  # [batch, output_dim]
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def _merge_bidir_hidden(self, hidden):
        """
        hidden: [num_layers*2, batch, hidden_per_dir]
        return: [num_layers, batch, hidden_per_dir*2]  (merge forward & backward)
        """
        num_layers_times_2, batch, hid_per_dir = hidden.size()
        num_layers = num_layers_times_2 // 2
        hidden = hidden.view(num_layers, 2, batch, hid_per_dir)  # [num_layers, 2, batch, hid_per_dir]
        # concat forward/back
        hidden = torch.cat([hidden[:,0,:,:], hidden[:,1,:,:]], dim=2)  # [num_layers, batch, hid_per_dir*2]
        return hidden

    def forward(self, src, src_lengths, trg=None, teacher_forcing=True):
        batch_size = src.size(0)
        trg_len = trg.size(1) if trg is not None else MAX_LEN_DECODING
        output_dim = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)  # encoder_outputs: [batch, src_len, hidden_dim]
        # merge bidirectional hidden -> for decoder init
        hidden_dec = self._merge_bidir_hidden(hidden)  # [num_layers, batch, hidden_dim]
        cell_dec = self._merge_bidir_hidden(cell)

        src_mask = (src != PAD_IDX_EN).to(self.device)

        input_token = trg[:,0] if trg is not None else torch.tensor([fr_vocab['<sos>']]*batch_size, device=self.device)
        for t in range(1, trg_len):
            teacher_force_flag = (random.random() < self.teacher_forcing_ratio) if (trg is not None and teacher_forcing) else False
            output, hidden_dec, cell_dec, _ = self.decoder(input_token, hidden_dec, cell_dec, encoder_outputs, src_mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = trg[:,t] if (teacher_force_flag and trg is not None) else top1
        return outputs


# --------------------
# Training / Eval
# --------------------
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


# --------------------
# Translation Helpers
# --------------------
def translate(sentence: str, model, device, max_len=MAX_LEN_DECODING):
    model.eval()
    src_tensor = encode_sentence(sentence, en_tokenizer, en_vocab).unsqueeze(0).to(device)
    src_lengths = torch.tensor([src_tensor.size(1)], dtype=torch.long).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lengths)

    # merge bidir
    def merge(hidden):
        num_layers_times_2, b, hidp = hidden.size()
        num_layers = num_layers_times_2 // 2
        h = hidden.view(num_layers, 2, b, hidp)
        return torch.cat([h[:,0,:,:], h[:,1,:,:]], dim=2)
    hidden_dec = merge(hidden)
    cell_dec = merge(cell)

    sos_idx = fr_vocab['<sos>']
    eos_idx = fr_vocab['<eos>']
    input_token = torch.tensor([sos_idx], dtype=torch.long, device=device)

    src_mask = (src_tensor != PAD_IDX_EN).to(device)

    output_ids = []
    for _ in range(max_len):
        with torch.no_grad():
            pred, hidden_dec, cell_dec, attn = model.decoder(input_token, hidden_dec, cell_dec, encoder_outputs, src_mask)
        top1 = pred.argmax(1).item()
        if top1 == eos_idx:
            break
        output_ids.append(top1)
        input_token = torch.tensor([top1], dtype=torch.long, device=device)

    # detokenize
    try:
        itos = fr_vocab.get_itos()
    except:
        itos = [tok for tok, idx in sorted(fr_vocab.get_stoi().items(), key=lambda x: x[1])]
    words = [itos[i] for i in output_ids if i < len(itos)]
    return " ".join(words)


def beam_translate(sentence: str, model, device, max_len=MAX_LEN_DECODING, beam_width=5, alpha=0.7):
    """
    Beam search with length penalty:
    score_normalized = log_prob_sum / ((5 + len)/6)^alpha
    """
    model.eval()
    src_tensor = encode_sentence(sentence, en_tokenizer, en_vocab).unsqueeze(0).to(device)
    src_lengths = torch.tensor([src_tensor.size(1)], device=device)
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lengths)

    # merge
    def merge(hidden):
        num_layers_times_2, b, hidp = hidden.size()
        num_layers = num_layers_times_2 // 2
        h = hidden.view(num_layers, 2, b, hidp)
        return torch.cat([h[:,0,:,:], h[:,1,:,:]], dim=2)
    hidden_dec = merge(hidden)
    cell_dec = merge(cell)

    sos_idx, eos_idx = fr_vocab['<sos>'], fr_vocab['<eos>']
    src_mask = (src_tensor != PAD_IDX_EN).to(device)

    # beam entries: (tokens_list, hidden, cell, log_prob_sum)
    beam = [([sos_idx], hidden_dec, cell_dec, 0.0)]
    completed = []

    for _step in range(max_len):
        new_beam = []
        for tokens, h, c, score in beam:
            if tokens[-1] == eos_idx:
                completed.append((tokens, score))
                continue
            input_token = torch.tensor([tokens[-1]], device=device)
            with torch.no_grad():
                pred, h_new, c_new, _ = model.decoder(input_token, h, c, encoder_outputs, src_mask)
                log_probs = F.log_softmax(pred, dim=1)  # [1, V]
            topk_logp, topk_idx = torch.topk(log_probs, beam_width, dim=1)
            for i in range(beam_width):
                idx_i = topk_idx[0, i].item()
                logp_i = topk_logp[0, i].item()
                new_beam.append((tokens + [idx_i], h_new, c_new, score + logp_i))
        # keep top-K by raw score
        beam = sorted(new_beam, key=lambda x: x[3], reverse=True)[:beam_width]
        if not beam:
            break

    # include leftover beams
    for tokens, h, c, score in beam:
        completed.append((tokens, score))

    # normalize with length penalty and pick best
    def norm_score(entry):
        tokens, s = entry
        length = len(tokens) - 1  # exclude <sos>
        lp = ((5 + length) / 6) ** alpha
        return s / lp

    best_tokens, _ = max(completed, key=norm_score)
    output_ids = best_tokens[1:]  # drop <sos>
    if eos_idx in output_ids:
        output_ids = output_ids[:output_ids.index(eos_idx)]
    words = [fr_vocab.get_itos()[i] for i in output_ids if i < len(fr_vocab.get_itos())]
    return " ".join(words)


# --------------------
# BLEU computation
# --------------------
def compute_bleu(model, src_file, trg_file, device, max_samples=None, use_beam=False, beam_width=5):
    smoothie = SmoothingFunction().method4
    scores = []
    with open(src_file, encoding='utf-8') as fsrc, open(trg_file, encoding='utf-8') as ftrg:
        for i, (sline, tline) in enumerate(zip(fsrc, ftrg)):
            if max_samples and i >= max_samples:
                break
            sline = sline.strip()
            tline = tline.strip()
            pred = beam_translate(sline, model, device, beam_width=beam_width) if use_beam else translate(sline, model, device)
            reference = [tline.split()]
            hypothesis = pred.split()
            score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
            scores.append(score)
    return sum(scores)/len(scores) if scores else 0.0


# --------------------
# Run training
# --------------------
def run_training():
    enc = Encoder(len(en_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_EN)
    dec = Decoder(len(fr_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_FR)
    model = Seq2Seq(enc, dec, DEVICE, TEACHER_FORCING_RATIO).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    # label smoothing to reduce overconfidence
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_FR, label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    train_loader = get_dataloader(EN_TRAIN, FR_TRAIN, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(EN_VAL, FR_VAL, batch_size=BATCH_SIZE, shuffle=False) if (os.path.exists(EN_VAL) and os.path.exists(FR_VAL)) else None

    best_valid_loss = float('inf')
    patience = 4
    wait = 0

    train_losses = []
    valid_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        start = time.time()

        # teacher forcing decay
        model.teacher_forcing_ratio = max(0.1, TEACHER_FORCING_RATIO * (0.95 ** (epoch - 1)))

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CLIP, DEVICE)
        valid_loss = evaluate(model, val_loader, criterion, DEVICE) if val_loader else train_loss
        end = time.time()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | TF: {model.teacher_forcing_ratio:.3f} | Time: {end-start:.1f}s")

        # Save best checkpoint
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            wait = 0
            print("  Saved new best model.")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(valid_loss)

    # plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('Loss_curve.png', dpi=300)
    plt.close()
    print("Saved Loss_curve.png")

    return model


if __name__ == "__main__":
    model = run_training()
    # load best model for return
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
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
            print("FR_pred:", beam_translate(s, model, DEVICE, beam_width=3))
            print("-"*30)

    if os.path.exists(EN_VAL) and os.path.exists(FR_VAL):
        bleu = compute_bleu(model, EN_VAL, FR_VAL, DEVICE, max_samples=200, use_beam=True, beam_width=5)
        print("BLEU on val:", bleu)