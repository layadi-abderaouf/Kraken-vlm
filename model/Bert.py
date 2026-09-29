import torch
import torch.nn as nn
from transformers import BertModel,logging
from transformers import BertTokenizer
import config


class BERTEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        logging.set_verbosity_error()

        

        self.bert = BertModel.from_pretrained(config.BERT_MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)

        # جمد جميع الطبقات
        for p in self.bert.parameters():
            p.requires_grad = False

        self.bert.eval()

    def forward(self, input_ids, attention_mask):
        
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            return outputs.last_hidden_state