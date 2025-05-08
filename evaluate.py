# evaluate.py

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import ToxicCommentsWithLabelsDataset
from bert_classifier import BertClassifier
from utils import calculate_metrics
import config
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import os


def eval_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

    return np.vstack(all_preds), np.vstack(all_labels)

def plot_confusion_matrix(labels, preds, output_path):
    mcm = multilabel_confusion_matrix(labels, preds)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        cm = mcm[i]
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Label {i}')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'confusion_matrices.png'))
    plt.close()

def plot_roc_curves(labels, probs, output_path):
    plt.figure(figsize=(10, 8))
    for i in range(labels.shape[1]):
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'roc_curves.png'))
    plt.close()

def plot_f1_per_class(labels, preds, output_path):
    from sklearn.metrics import f1_score
    f1s = f1_score(labels, preds, average=None)
    plt.bar(range(len(f1s)), f1s)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Per Class')
    plt.xticks(range(len(f1s)))
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_path, 'f1_per_class.png'))
    plt.close()

def training_graph():
    epochs = [1, 2, 3]
    train_loss = [0.5116, 0.2827, 0.2338]
    val_f1 = [0.6354, 0.6667, 0.6667]
    val_acc = [0.9500, 0.9629, 0.9629]

    epochs = [1, 2, 3]
    train_loss = [0.5116, 0.2827, 0.2338]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='o', color='tab:red', label='Train Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_loss.png')
    plt.show()


def main():
    training_graph()
    print(stop)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)

    test_dataset = ToxicCommentsWithLabelsDataset(
        config.TEST_CSV_PATH,
        config.TEST_LABELS_CSV_PATH,
        tokenizer,
        config.MAX_LEN
    )
    test_loader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE)

    model = BertClassifier().to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()

    preds, labels = eval_model(model, test_loader, device)
    binary_preds = (preds >= 0.5).astype(int)

    # Output path
    output_path = "evaluation_outputs"
    os.makedirs(output_path, exist_ok=True)

    # Plotting
    plot_confusion_matrix(labels, binary_preds, output_path)
    plot_roc_curves(labels, preds, output_path)
    plot_f1_per_class(labels, binary_preds, output_path)

    print("Evaluation complete. Charts saved to", output_path)


if __name__ == "__main__":
    main()
