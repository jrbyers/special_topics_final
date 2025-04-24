import torch.nn as nn
from transformers import BertModel
import config

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(config.LABEL_COLUMNS))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)