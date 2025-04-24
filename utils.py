from sklearn.metrics import f1_score, accuracy_score
import torch

def calculate_metrics(outputs, targets):
    # Apply sigmoid activation to the outputs to get probabilities
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    # Convert the outputs to binary predictions based on a threshold (0.5)
    preds = (outputs >= 0.5).astype(int)

    # Check the shape of the outputs and targets to ensure they are in a compatible format
    assert preds.shape == targets.shape, f"Shape mismatch: {preds.shape} vs {targets.shape}"

    # Compute F1 score and accuracy for multi-label classification
    f1 = f1_score(targets, preds, average='macro', zero_division=1)  # Added zero_division=1
    acc = accuracy_score(targets, preds)  # Calculate accuracy as the proportion of correct predictions

    return f1, acc
