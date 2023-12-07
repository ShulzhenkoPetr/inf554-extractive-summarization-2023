import json
from pathlib import Path
import tqdm
import numpy as np

from torch.utils.data import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer

from utils.utils import log_results, to_lower_case
from utils.custom_datasets import UtterancesDataset
from utils.metrics import accuracy_metric, f1_metric


def sbert_baseline():

    train_dataset = UtterancesDataset(
                                        data_folder='dataset/all_training_data/train/',
                                        label_path='dataset/all_training_data/train_labels.json',
                                        exception_files=['IS1002a', 'IS1005d', 'TS3012c'],
                                        preproc=to_lower_case
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = UtterancesDataset(
                                        data_folder='dataset/all_training_data/val/',
                                        label_path='dataset/all_training_data/val_labels.json',
                                        exception_files=['IS1002a', 'IS1005d', 'TS3012c'],
                                        preproc=to_lower_case
    )
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    train_sbert_embeddings = []
    train_labels = []

    for utts, labels in tqdm.tqdm(train_dataloader, desc='train embeddings'):
        train_sbert_embeddings.extend(sbert.encode(utts))
        train_labels.extend(labels)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_sbert_embeddings, train_labels)

    val_labels = []
    val_predictions = []
    for utts, labels in tqdm.tqdm(val_dataloader, desc='val embeddings'):
        embs = sbert.encode(utts)
        preds = clf.predict(embs)

        val_predictions.extend(preds)
        val_labels.extend(labels)

    accuracy = accuracy_metric(val_predictions, val_labels)
    f1_score = f1_metric(val_predictions, val_labels)

    print(val_predictions)

    print("Val accuracy", accuracy)
    print("Val f1 score", f1_score)

    log_results('logs/sbert_baseline.txt',
                'default sbert model to_lower_case with DecisionTree',
                accuracy, f1_score)


if __name__ == '__main__':
    sbert_baseline()





