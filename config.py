PRETRAINED_MODEL_NAME = "bert-base-uncased"

TRAIN_CSV_PATH        = "data/train.csv"
TEST_CSV_PATH         = "data/test.csv"
TEST_LABELS_CSV_PATH  = "data/test_labels.csv"   # if you use a separate labels file

LABEL_COLUMNS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

# cover longer comments
MAX_LEN = 192

# original was 16; we'll accumulate to simulate 32
TRAIN_BATCH_SIZE  = 16
VALID_BATCH_SIZE  = 32

# stretch to up to 10 epochs, but early‑stop on F1
EPOCHS = 10
PATIENCE = 3

LEARNING_RATE     = 2e-5
WEIGHT_DECAY      = 0.01
WARMUP_PROPORTION = 0.10   # 10% of total steps

# accumulation to simulate 32→64 as you like
GRAD_ACCUM_STEPS = 2

MODEL_PATH = "checkpoints/bert_finetuned.bin"
