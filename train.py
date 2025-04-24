import torch
from transformers import BertTokenizer, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import ToxicCommentsDataset, ToxicCommentsWithLabelsDataset
from bert_classifier import BertClassifier
from utils import calculate_metrics
import config
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_model(model, dataloader, device):
    model.eval()
    total_f1, total_acc = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            f1, acc = calculate_metrics(outputs, labels)
            total_f1 += f1
            total_acc += acc

    return total_f1 / len(dataloader), total_acc / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)

    train_dataset = ToxicCommentsDataset(config.TRAIN_CSV_PATH, tokenizer, config.MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)

    test_dataset = ToxicCommentsWithLabelsDataset(config.TEST_CSV_PATH, config.TEST_LABELS_CSV_PATH, tokenizer, config.MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE)

    model = BertClassifier().to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    num_training_steps = len(train_loader) * config.EPOCHS
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_f1, val_acc = eval_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val F1 = {val_f1:.4f}, Val Acc = {val_acc:.4f}")

    torch.save(model.state_dict(), config.MODEL_PATH)
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()