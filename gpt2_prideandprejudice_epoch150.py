"""
GPT-style character-level language model trained on Pride and Prejudice.
"""

import math
from pathlib import Path
import urllib.request
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Configuration
# ----------------------------

DATA_URL = "https://www.gutenberg.org/ebooks/1342.txt.utf-8"
DATA_FILE = "pride_and_prejudice.txt"

TRAINING_LOG_FILE = "training_log_150.csv"
GENERATED_TEXT_FILE = "generated_text_150.txt"
CHECKPOINT_FILE = "gpt2_pride_checkpoint.pt"

BLOCK_SIZE = 64
BATCH_SIZE = 64
MAX_EPOCHS = 150
MAX_STEPS_PER_EPOCH = 300
LEARNING_RATE = 3e-4

EMB_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

EVAL_STEPS = 50
START_TEXT = "It is "
MAX_NEW_TOKENS = 500
SEED = 2026


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_data_if_needed():
    if not Path(DATA_FILE).exists():
        print("[INFO] Downloading Pride and Prejudice dataset from Project Gutenberg...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)


def strip_gutenberg_header_footer(text):
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***",
        "*** START OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***",
        "*** END OF THIS PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = idx + len(marker)
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def load_data():
    download_data_if_needed()

    text = open(DATA_FILE, "r", encoding="utf-8").read()
    text = strip_gutenberg_header_footer(text)

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"[INFO] total characters: {len(text):,}")
    print(f"[INFO] vocab size: {len(chars)}")
    print(f"[INFO] train tokens: {len(train_data):,}")
    print(f"[INFO] val tokens: {len(val_data):,}")

    return train_data, val_data, stoi, itos, len(chars)


class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, emb_dim, block_size):
        super().__init__()
        pe = torch.zeros(block_size, emb_dim)
        pos = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, T):
        return self.pe[:T]


class Head(nn.Module):
    def __init__(self, emb_dim, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([
            Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ffwd = FeedForward(emb_dim, dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size,
                 emb_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(emb_dim, block_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_embedding(x)
        pos = self.pos_encoding(T)[None]
        h = self.drop(tok + pos)
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits


def sequence_cross_entropy(logits, targets):
    return F.cross_entropy(logits.transpose(1, 2), targets)


def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    total_loss, total_count = 0.0, 0

    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

        if max_steps is not None and step + 1 >= max_steps:
            break

    return total_loss / total_count


@torch.no_grad()
def evaluate_loss(model, loader, device, max_steps=None):
    model.eval()
    total_loss, total_count = 0.0, 0

    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)

        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

        if max_steps is not None and step + 1 >= max_steps:
            break

    return total_loss / total_count


@torch.no_grad()
def sample_gpt(model, block_size, stoi, itos, device,
               start_text=START_TEXT, max_new_tokens=MAX_NEW_TOKENS):
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)

    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)

    out = list(start_text)

    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)

    return "".join(out)


def save_training_log(log_rows):
    with open(TRAINING_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log_rows)


def main():
    set_seed()
    device = get_device()

    block_size = BLOCK_SIZE
    train_data, val_data, stoi, itos, vocab_size = load_data()

    train_dataset = NextTokenDataset(train_data, block_size)
    val_dataset = NextTokenDataset(val_data, block_size)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GPT(
        vocab_size,
        block_size,
        emb_dim=EMB_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"[INFO] device: {device}")
    print(f"[INFO] model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("[INFO] training started")

    log_rows = []
    best_val_loss = float("inf")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            max_steps=MAX_STEPS_PER_EPOCH
        )
        val_loss = evaluate_loss(
            model, val_loader, device,
            max_steps=EVAL_STEPS
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
        })

        print(f"epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "model_state_dict": model.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "config": {
                    "block_size": BLOCK_SIZE,
                    "emb_dim": EMB_DIM,
                    "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "vocab_size": vocab_size,
                },
            }, CHECKPOINT_FILE)

    save_training_log(log_rows)

    generated_text = sample_gpt(
        model, block_size, stoi, itos, device,
        start_text=START_TEXT,
        max_new_tokens=MAX_NEW_TOKENS
    )

    with open(GENERATED_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(generated_text)

    print(f"[INFO] training log saved to {TRAINING_LOG_FILE}")
    print(f"[INFO] generated text saved to {GENERATED_TEXT_FILE}")
    print(f"[INFO] best checkpoint saved to {CHECKPOINT_FILE}")
    print("\n===== Generated Sample =====\n")
    print(generated_text)


if __name__ == "__main__":
    main()
