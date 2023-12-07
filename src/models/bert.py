import torch
from torch import nn

from transformers import BertModel


class Bert(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 250),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(250, 2)
        )

    def forward(self, input_ids, attention_mask, ):
        _, pool_output = self.encoder(input_ids,
                                      attention_mask=attention_mask,
                                      # token_type_ids=token_type_ids,
                                      return_dict=False)
        return self.mlp(pool_output)

