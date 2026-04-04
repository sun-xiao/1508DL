"""
Hyperparameter tuning for TextLSTM on 5% of training data.
Grid search over learning rate, hidden_dim, and dropout.
"""

import itertools
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from vocab import Vocab
from cnn_dataset import TextDataset
from text_lstm import TextLSTM


# -------------------------
# Fixed config
# -------------------------
TRAIN_PATH = "./data/processed_v2/train.csv"
RANDOM_SEED = 2026
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
MAX_LEN = 40
EMBED_DIM = 128
NUM_LAYERS = 2
BIDIRECTIONAL = True
BATCH_SIZE = 256
GRAD_CLIP = 5.0
EPOCHS = 5
DATA_FRACTION = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Search space
# -------------------------
PARAM_GRID = {
    "lr":         [5e-4, 1e-3, 2e-3],
    "hidden_dim": [128, 256, 512],
    "dropout":    [0.3, 0.5, 0.7],
}


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(input_ids), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids)
        total_loss += criterion(logits, labels).item() * input_ids.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, f1


def run_trial(params, train_loader, val_loader, vocab):
    set_seed(RANDOM_SEED)

    model = TextLSTM(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=params["hidden_dim"],
        num_layers=NUM_LAYERS,
        dropout=params["dropout"],
        bidirectional=BIDIRECTIONAL,
        pad_idx=vocab.pad_idx,
    ).to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, GRAD_CLIP)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_f1)
        best_val_f1 = max(best_val_f1, val_f1)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return best_val_f1, num_params


def main():
    print(f"Using device: {DEVICE}")
    print(f"Data fraction: {DATA_FRACTION * 100:.0f}%\n")

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df = train_df.dropna(subset=["text", "target"]).reset_index(drop=True)

    # Sample 5% stratified
    train_df, _ = train_test_split(
        train_df, train_size=DATA_FRACTION,
        random_state=RANDOM_SEED, stratify=train_df["target"],
    )
    train_df = train_df.reset_index(drop=True)

    # Split into train/val
    train_part, val_part = train_test_split(
        train_df, test_size=0.2,
        random_state=RANDOM_SEED, stratify=train_df["target"],
    )

    X_train, y_train = train_part["text"].tolist(), train_part["target"].tolist()
    X_val, y_val = val_part["text"].tolist(), val_part["target"].tolist()

    print(f"Tune train size: {len(X_train)}")
    print(f"Tune val size  : {len(X_val)}")

    print("Building vocabulary...")
    vocab = Vocab(texts=X_train, max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ)
    print(f"Vocab size: {len(vocab)}\n")

    train_dataset = TextDataset(X_train, y_train, vocab, max_len=MAX_LEN)
    val_dataset = TextDataset(X_val, y_val, vocab, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Grid search
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total = len(combos)

    print(f"Running {total} trials ({EPOCHS} epochs each)...")
    print("=" * 75)

    results = []
    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n[{i}/{total}] {label}")

        best_f1, num_params = run_trial(params, train_loader, val_loader, vocab)
        results.append({**params, "val_f1": best_f1, "params": num_params})
        print(f"  -> val F1: {best_f1:.4f}  ({num_params:,} parameters)")

    # Sort and display results
    results.sort(key=lambda r: r["val_f1"], reverse=True)

    print("\n" + "=" * 75)
    print("RESULTS (sorted by val F1)")
    print("=" * 75)
    print(f"{'Rank':<5} {'LR':<10} {'Hidden':<10} {'Dropout':<10} {'Val F1':<10} {'Params':<15}")
    print("-" * 60)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5} {r['lr']:<10.4f} {r['hidden_dim']:<10} {r['dropout']:<10.1f} {r['val_f1']:<10.4f} {r['params']:<15,}")

    best = results[0]
    print(f"\nBest config: lr={best['lr']}, hidden_dim={best['hidden_dim']}, dropout={best['dropout']}")
    print(f"Best val F1: {best['val_f1']:.4f}")


if __name__ == "__main__":
    main()
