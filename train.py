import os
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from src.model import Transformer
from src.dataset import build_vocab, get_dataloaders
from src.vocabulary import Vocabulary
from src.bleu import compute_bleu

os.makedirs('Models', exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 128
EPOCHS      = 30
D_MODEL     = 256
N_LAYERS    = 3
N_HEADS     = 8
D_FF        = 512
DROPOUT     = 0.4
MAX_LEN     = 64
WARMUP_STEPS = 400
BLEU_MAX_LEN = 64  # max generation length for BLEU eval

# ── Vocabularies & data ───────────────────────────────────────────────────────
print('Building vocabularies...')
src_vocab, tgt_vocab = build_vocab(min_freq=2)
print(f'  src vocab: {len(src_vocab):,}  |  tgt vocab: {len(tgt_vocab):,}')

print('Creating dataloaders...')
train_loader, val_loader, test_loader = get_dataloaders(src_vocab, tgt_vocab, BATCH_SIZE)

PAD_IDX = tgt_vocab.word2idx[Vocabulary.PAD]

# ── Model ─────────────────────────────────────────────────────────────────────
model = Transformer(
    source_vocab_size=len(src_vocab),
    target_vocab_size=len(tgt_vocab),
    d_model=D_MODEL,
    N=N_LAYERS,
    heads=N_HEADS,
    d_ff=D_FF,
    drop_rate=DROPOUT,
    max_len=MAX_LEN,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  model parameters: {n_params:,}')

optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

def get_lr_schedule(step):
    step = max(1, step)
    return (D_MODEL ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))

scheduler = LambdaLR(optimizer, lr_lambda=get_lr_schedule)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

# ── Mask helpers ──────────────────────────────────────────────────────────────
def make_src_mask(src):
    # (B, 1, 1, S)  —  1 where token is real, 0 where padding
    return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt):
    B, T = tgt.shape
    pad_mask    = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)          # (B, 1, 1, T)
    causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool()  # (T, T)
    return pad_mask & causal_mask                                      # (B, 1, T, T)


# ── Train / eval loop ─────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            tgt_input = tgt[:, :-1]   # drop <eos>  — decoder input
            tgt_label = tgt[:, 1:]    # drop <sos>  — ground truth

            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask)  # (B, T-1, V)

            loss = criterion(
                logits.reshape(-1, len(tgt_vocab)),
                tgt_label.reshape(-1),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()

    return total_loss / len(loader)


# ── Training ──────────────────────────────────────────────────────────────────
writer = SummaryWriter('runs/transformer')
train_losses, val_losses = [], []

print(f'\nTraining on {DEVICE} for {EPOCHS} epochs\n')
for epoch in range(1, EPOCHS + 1):
    epoch_start = datetime.now()
    train_loss = run_epoch(train_loader, train=True)
    val_loss   = run_epoch(val_loader,   train=False)
    val_bleu   = compute_bleu(model, val_loader, tgt_vocab, BLEU_MAX_LEN, DEVICE)
    elapsed = (datetime.now() - epoch_start).total_seconds()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('BLEU/val', val_bleu, epoch)
    writer.add_scalar('Learning_rate', current_lr, epoch)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] Epoch {epoch:03d}/{EPOCHS}  |  {elapsed:6.1f}s  |  train {train_loss:.4f}  |  val {val_loss:.4f}  |  val BLEU {val_bleu:.2f}')

    if epoch % 10 == 0 or epoch == EPOCHS:
        torch.save(model.state_dict(), f'Models/checkpoint_epoch{epoch:03d}.pt')

test_loss = run_epoch(test_loader, train=False)
test_bleu = compute_bleu(model, test_loader, tgt_vocab, BLEU_MAX_LEN, DEVICE)
print(f'\nTest loss: {test_loss:.4f}  |  Test BLEU: {test_bleu:.2f}')

writer.add_scalar('Loss/test', test_loss, 0)
writer.add_scalar('BLEU/test', test_bleu, 0)
writer.close()

# ── Loss plot ─────────────────────────────────────────────────────────────────
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker='o', label='Train')
plt.plot(epochs, val_losses,   marker='o', label='Val')
plt.axhline(test_loss, color='r', linestyle='--', label=f'Test ({test_loss:.4f})')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Transformer — En→De (Multi30k)')
plt.legend()
plt.tight_layout()
plt.savefig('images/losses.png', dpi=150)
print('Loss plot saved to images/losses.png')
