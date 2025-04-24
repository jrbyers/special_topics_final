PRETRAINED_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_PATH = "best_model.pt"
LABEL_COLUMNS = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]

TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"
TEST_LABELS_CSV_PATH = "data/test_labels.csv"