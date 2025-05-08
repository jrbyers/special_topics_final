import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from torch.optim import AdamW
import config
from dataset import ToxicCommentsDataset
from evaluate import calculate_metrics
from bert_classifier import BertClassifier
 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
GRAD_ACCUM_STEPS = 1
PATIENCE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
def load_data():
    train_df = pd.read_csv(config.TRAIN_CSV_PATH)
    val_df   = pd.read_csv(config.TEST_CSV_PATH)
    return train_df, val_df
 
 
def create_dataloaders(train_df, val_df, tokenizer):
    train_ds = ToxicCommentsDataset(
        texts=train_df.comment_text.to_numpy(),
        labels=train_df[config.LABEL_COLUMNS].values,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    val_ds = ToxicCommentsDataset(
        texts=val_df.comment_text.to_numpy(),
        labels=val_df[config.LABEL_COLUMNS].values,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
 
    train_loader = DataLoader(
        train_ds,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader
 
 
def train():
    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    train_df, val_df = load_data()
    train_loader, val_loader = create_dataloaders(train_df, val_df, tokenizer)
 
    model = BertClassifier().to(DEVICE)
 
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.01
    )
 
    total_steps = len(train_loader) * config.EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = int(0.1 * total_steps)
 
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
 
    best_f1 = 0.0
    no_improve = 0
 
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
 
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{config.EPOCHS} [TRAIN]")
        optimizer.zero_grad()
 
        for step, batch in pbar:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets        = batch["targets"].to(DEVICE).float()
 
            outputs = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(outputs, targets)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
 
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
 
            running_loss += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix(loss=running_loss / (step + 1))
 
        model.eval()
        all_outputs, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [VAL]"):
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                targets        = batch["targets"].to(DEVICE)
 
                outputs = model(input_ids, attention_mask)
                all_outputs.append(outputs)
                all_targets.append(targets)
 
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        val_f1, val_acc = calculate_metrics(all_outputs, all_targets)
 
        print(f"Validation F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
 
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"New best model saved (F1 {best_f1:.4f})")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s)")
 
        if no_improve >= PATIENCE:
            print("Early stopping triggered")
            break
 
 
if __name__ == "__main__":
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    train()