import torch
import torch.nn as nn
from transformers import BertModel

class FakeNewsClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(FakeNewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        return self.out(self.drop(pooled_output))
