PRETRAINED_MODEL_NAME = "bert-base-uncased"

TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"
TEST_LABELS_CSV_PATH  = "data/test_labels.csv"

LABEL_COLUMNS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

MAX_LEN = 192
TRAIN_BATCH_SIZE  = 16
VALID_BATCH_SIZE  = 32
EPOCHS = 10
PATIENCE = 3

LEARNING_RATE     = 2e-5
WEIGHT_DECAY      = 0.01
WARMUP_PROPORTION = 0.10

GRAD_ACCUM_STEPS = 2
MODEL_PATH = "best_model.pt"
