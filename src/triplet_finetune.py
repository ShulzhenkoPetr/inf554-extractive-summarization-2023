import json
from pathlib import Path

import torch.cuda
import tqdm
import numpy as np

import wandb

from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertTokenizer
from transformers import TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample

from models.bert import Bert
from utils.utils import log_results, to_lower_case
from utils.metrics import accuracy_metric, f1_metric

from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from utils.custom_datasets import triplet_data, sentences_data


model = SentenceTransformer('all-MiniLM-L6-v2')
train_data = triplet_data(data_folder='dataset/all_training_data/train/',
                                label_path='dataset/all_training_data/train_labels.json',
                                exception_files=['IS1002a', 'IS1005d', 'TS3012c'])
train_dataset = SentencesDataset(train_data, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model=model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3,
          warmup_steps=100, output_path='models/sbert_finetune/3_epochs/')


train_sentences, train_labels = sentences_data(data_folder='dataset/all_training_data/train/',
                                           label_path='dataset/all_training_data/train_labels.json',
                                           exception_files=['IS1002a', 'IS1005d', 'TS3012c'])
enc_train_sentences = model.encode(train_sentences, show_progress_bar=True)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(enc_train_sentences, train_labels)

rf = RandomForestClassifier(random_state=0)
rf.fit(enc_train_sentences, train_labels)


val_sentences, val_labels = sentences_data(data_folder='dataset/all_training_data/val/',
                                           label_path='dataset/all_training_data/val_labels.json',
                                           exception_files=['IS1002a', 'IS1005d', 'TS3012c'])

enc_val_sentences = model.encode(val_sentences, show_progress_bar=True)
val_predictions = clf.predict(enc_val_sentences)
rf_preds = rf.predict(enc_val_sentences)

accuracy = accuracy_metric(val_predictions, val_labels)
f1_score = f1_metric(val_predictions, val_labels)

rf_acc = accuracy_metric(rf_preds, val_labels)
rf_f1 = f1_metric(rf_preds, val_labels)

print(val_predictions)

print("Val accuracy", accuracy)
print("Val f1 score", f1_score)
print("RF val accuracy", rf_acc)
print("RF val f1 score", rf_f1)

log_results('logs/sbert_finetune.txt',
            'finetuned sbert model with DecisionTree',
            accuracy, f1_score)