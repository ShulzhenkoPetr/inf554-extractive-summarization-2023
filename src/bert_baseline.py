import json
from pathlib import Path

import torch.cuda
import tqdm
import numpy as np

from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import TrainingArguments, Trainer

from models.bert import Bert
from utils.utils import log_results, to_lower_case
from utils.custom_datasets import UtterancesBertDataset
from utils.metrics import accuracy_metric, f1_metric


def evaluate_bert():
    pass


def finetune_bert():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = UtterancesBertDataset(
                                        data_folder='dataset/all_training_data/train/',
                                        label_path='dataset/all_training_data/train_labels.json',
                                        exception_files=['IS1002a', 'IS1005d', 'TS3012c'],
                                        tokenizer=tokenizer,
                                        max_len=512
    )
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = UtterancesBertDataset(
                                        data_folder='dataset/all_training_data/val/',
                                        label_path='dataset/all_training_data/val_labels.json',
                                        exception_files=['IS1002a', 'IS1005d', 'TS3012c'],
                                        tokenizer=tokenizer,
                                        max_len=512
    )
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model_name = 'bert-base-uncased'
    bert = Bert(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert.to(device)

    args = TrainingArguments(
        f"bert-finetune-pool-mlp",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    trainer = Trainer(
        bert,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=f1_metric
    )

    trainer.train()


if __name__ == '__main__':
    finetune_bert()





