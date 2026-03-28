import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from vocab import Vocab
from cnn_dataset import TextDataset
from text_lstm import TextLSTM


# -------------------------
# Config
# -------------------------
TRAIN_PATH = "./data/processed_v2/train.csv"
TEST_PATH = "./data/processed_v2/test.csv"
CHECKPOINT_PATH = "./checkpoints/lstm_best.pt"

RANDOM_SEED = 2026
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
MAX_LEN = 40

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

BATCH_SIZE = 256
LR = 2e-3
EPOCHS = 5
GRAD_CLIP = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Metrics
# -------------------------
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# -------------------------
# Train / Eval Loops
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_preds = []

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        total_loss += loss.item() * input_ids.size(0)

        all_labels.extend(labels.cpu().numpy().astype(int).tolist())
        all_preds.extend(preds.cpu().numpy().astype(int).tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics, all_labels, all_preds


# -------------------------
# Main
# -------------------------
def main():
    set_seed(RANDOM_SEED)

    print("Using device:", DEVICE)

    # Step 1: Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = train_df.dropna(subset=["text", "target"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["text", "target"]).reset_index(drop=True)

    # Step 2: Split train into train/val
    print("Creating validation split from train.csv...")
    train_part, val_part = train_test_split(
        train_df,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=train_df["target"],
    )

    X_train = train_part["text"].tolist()
    y_train = train_part["target"].tolist()

    X_val = val_part["text"].tolist()
    y_val = val_part["target"].tolist()

    X_test = test_df["text"].tolist()
    y_test = test_df["target"].tolist()

    # temporary
    # X_train = X_train[:50000]
    # y_train = y_train[:50000]
    # X_val = X_val[:10000]
    # y_val = y_val[:10000]
    X_test = X_test[:10000]
    y_test = y_test[:10000]

    print(f"Train size: {len(X_train)}")
    print(f"Val size  : {len(X_val)}")
    print(f"Test size : {len(X_test)}")

    # Step 3: Build vocab from training texts only
    print("Building vocabulary...")
    vocab = Vocab(
        texts=X_train,
        max_size=MAX_VOCAB_SIZE,
        min_freq=MIN_FREQ,
    )
    print("Vocab size:", len(vocab))

    # Step 4: Datasets / Loaders
    train_dataset = TextDataset(X_train, y_train, vocab, max_len=MAX_LEN)
    val_dataset = TextDataset(X_val, y_val, vocab, max_len=MAX_LEN)
    test_dataset = TextDataset(X_test, y_test, vocab, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Step 5: Build model
    model_config = dict(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        pad_idx=vocab.pad_idx,
    )

    print("Building TextLSTM model...")
    model = TextLSTM(**model_config).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    criterion = torch.nn.BCEWithLogitsLoss()

    # Step 6: Train or load checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\nCheckpoint found at {CHECKPOINT_PATH}, skipping training.")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from checkpoint (val F1: {checkpoint.get('best_val_f1', 'N/A')})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        best_val_f1 = -1.0
        best_state_dict = None

        print("Training TextLSTM...")
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, GRAD_CLIP)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}/{EPOCHS}  (lr={current_lr:.6f})")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss  : {val_loss:.4f}")
            print(f"Val Acc   : {val_metrics['accuracy']:.4f}")
            print(f"Val Prec  : {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1    : {val_metrics['f1']:.4f}")

            scheduler.step(val_metrics["f1"])

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print("\nLoading best model based on validation F1...")
        model.load_state_dict(best_state_dict)

        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save({
            "model_state_dict": best_state_dict,
            "model_config": model_config,
            "vocab_stoi": vocab.stoi,
            "vocab_itos": vocab.itos,
            "best_val_f1": best_val_f1,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    # Step 7: Evaluate on test
    print("Evaluating on test set...")
    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

    print("\nTest Results")
    print("Test Loss :", round(test_loss, 4))
    print("Accuracy  :", round(test_metrics["accuracy"], 4))
    print("Precision :", round(test_metrics["precision"], 4))
    print("Recall    :", round(test_metrics["recall"], 4))
    print("F1 Score  :", round(test_metrics["f1"], 4))

    print("\nDetailed report:")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
