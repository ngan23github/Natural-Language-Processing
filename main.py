#!/usr/bin/env python
# coding: utf-8

# ## Cài đặt môi trường

# ### Import thư viện

# In[22]:


import os
import random
import time
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ### Config

# In[23]:


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


# In[24]:


print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))


# ### Tokenization

# In[25]:


en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')


# ### Xây dựng từ điển

# In[26]:


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

# adjust data paths if needed
EN_TRAIN = './data/train.en'
FR_TRAIN = './data/train.fr'
EN_VAL = './data/val.en'
FR_VAL = './data/val.fr'

en_vocab = build_vocab(EN_TRAIN, en_tokenizer)
fr_vocab = build_vocab(FR_TRAIN, fr_tokenizer)

PAD_IDX_EN = en_vocab['<pad>']
PAD_IDX_FR = fr_vocab['<pad>']


# ### Encode sentence

# In[27]:


def encode_sentence(sentence, tokenizer, vocab):
    tokens = tokenizer(sentence.strip().lower())
    ids = [vocab['<sos>']] + [vocab[token] for token in tokens] + [vocab['<eos>']]
    return torch.tensor(ids, dtype=torch.long)


# ### Dataset

# In[28]:


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


# ### Collate fn với Padding & Packing

# In[29]:


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


# ### DataLoader

# In[30]:


def get_dataloader(src_file, trg_file, batch_size=BATCH_SIZE, shuffle=True):
    dataset = TranslationDataset(src_file, trg_file, en_tokenizer, fr_tokenizer, en_vocab, fr_vocab)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


# ## Models

# In[31]:


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

    def forward(self, src, src_lengths):
        # src: [batch, src_len]
        embedded = self.embedding(src)  # [batch, src_len, embed_dim]
        packed_emb = pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, (hidden, cell) = self.lstm(packed_emb)
        return hidden, cell


# In[32]:


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden, cell):
        # input_token: [batch]
        input_token = input_token.unsqueeze(1)            # [batch,1]
        embedded = self.embedding(input_token)            # [batch,1,embed_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))       # [batch, output_dim]
        return prediction, hidden, cell


# In[33]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, src_lengths, trg=None, teacher_forcing=True):
        """
        If trg provided and teacher_forcing=True, do training-mode decoding using trg.
        If trg None or teacher_forcing False -> do greedy decoding up to trg.size(1) or MAX.
        """
        batch_size = src.size(0)
        # if trg provided length else use max decode length
        if trg is not None:
            trg_len = trg.size(1)
        else:
            trg_len = MAX_LEN_DECODING

        output_dim = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        hidden, cell = self.encoder(src, src_lengths)

        # first input = <sos> from trg if available else build tensor
        if trg is not None:
            input_token = trg[:, 0]
        else:
            sos_idx = fr_vocab['<sos>']
            input_token = torch.tensor([sos_idx] * batch_size, dtype=torch.long, device=self.device)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output

            if trg is not None and teacher_forcing:
                teacher_force = random.random() < self.teacher_forcing_ratio
            else:
                teacher_force = False

            top1 = output.argmax(1)
            if teacher_force and trg is not None:
                input_token = trg[:, t]
            else:
                input_token = top1

        return outputs


# ## Huấn luyện:

# In[34]:


def train_one_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for src, trg, src_lengths, trg_lengths in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        src_lengths = src_lengths.to(device)

        optimizer.zero_grad()

        output = model(src, src_lengths, trg, teacher_forcing=True)
        # output: [batch, trg_len, vocab_size]
        output_dim = output.shape[-1]

        # shift to ignore first token (<sos>)
        output = output[:, 1:, :].reshape(-1, output_dim)
        trg_y = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# In[35]:


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg, src_lengths, trg_lengths in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            src_lengths = src_lengths.to(device)

            # disable teacher forcing
            output = model(src, src_lengths, trg, teacher_forcing=False)
            output_dim = output.shape[-1]

            output = output[:, 1:, :].reshape(-1, output_dim)
            trg_y = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg_y)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# In[36]:


def translate(sentence: str, model, device, max_len=MAX_LEN_DECODING):
    model.eval()
    src_tensor = encode_sentence(sentence, en_tokenizer, en_vocab).unsqueeze(0).to(device)
    src_lengths = torch.tensor([src_tensor.size(1)], dtype=torch.long).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_lengths)

    sos_idx = fr_vocab['<sos>']
    eos_idx = fr_vocab['<eos>']
    input_token = torch.tensor([sos_idx], dtype=torch.long, device=device)

    output_ids = []
    for _ in range(max_len):
        with torch.no_grad():
            pred, hidden, cell = model.decoder(input_token, hidden, cell)
        top1 = pred.argmax(1).item()
        if top1 == eos_idx:
            break
        output_ids.append(top1)
        input_token = torch.tensor([top1], dtype=torch.long, device=device)

    # detokenize
    try:
        itos = fr_vocab.get_itos()
    except:
        # fallback for older versions
        itos = [tok for tok, idx in sorted(fr_vocab.get_stoi().items(), key=lambda x: x[1])]
    words = [itos[i] for i in output_ids]
    return " ".join(words)


# ## Đánh giá BLEU:

# In[37]:


def compute_bleu(model, src_file, trg_file, device, max_samples=None):
    smoothie = SmoothingFunction().method4
    scores = []
    with open(src_file, encoding='utf-8') as fsrc, open(trg_file, encoding='utf-8') as ftrg:
        for i, (sline, tline) in enumerate(zip(fsrc, ftrg)):
            if max_samples and i >= max_samples:
                break
            sline = sline.strip()
            tline = tline.strip()
            pred = translate(sline, model, device)
            reference = [tline.split()]
            hypothesis = pred.split()
            score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
            scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# ## Dịch thử:

# In[38]:


def run_training():
    enc = Encoder(len(en_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_EN)
    dec = Decoder(len(fr_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_FR)
    model = Seq2Seq(enc, dec, DEVICE, teacher_forcing_ratio=TEACHER_FORCING_RATIO).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX_FR)

    train_loader = get_dataloader(EN_TRAIN, FR_TRAIN, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(EN_VAL, FR_VAL, batch_size=BATCH_SIZE, shuffle=False) if os.path.exists(EN_VAL) else None

    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    patience = 3
    wait = 0

    for epoch in range(1, N_EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CLIP, DEVICE)
        valid_loss = evaluate(model, val_loader, criterion, DEVICE) if val_loader else train_loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end = time.time()
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Time: {end-start:.1f}s")

        # save checkpoint if improved
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


# In[39]:


if __name__ == "__main__":
    # train (or you can load checkpoint)
    model = run_training()

    # load best model if exists:
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=DEVICE)
        # must recreate model architecture then load
        enc = Encoder(len(en_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_EN)
        dec = Decoder(len(fr_vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PAD_IDX_FR)
        model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
        model.load_state_dict(data['model_state_dict'])
        print("Loaded best checkpoint.")

    # sample tests (thay bằng câu bạn muốn test)
    samples = [
        "I am going to school.",
        "How are you today?",
        "The bus is late.",
        "We need to finish the project.",
        "This is a simple sentence to translate."
    ]
    for s in samples:
        print("EN:", s)
        print("FR_pred:", translate(s, model, DEVICE))
        print("-" * 30)

    # compute BLEU on a small subset of validation if available
    if os.path.exists(EN_VAL) and os.path.exists(FR_VAL):
        bleu = compute_bleu(model, EN_VAL, FR_VAL, DEVICE, max_samples=200)
        print("BLEU on val (avg):", bleu)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script main.ipynb')


# In[ ]:




