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

from models.bert import Bert
from utils.utils import log_results, to_lower_case
from utils.custom_datasets import UtterancesBertDataset
from utils.metrics import accuracy_metric, f1_metric


def evaluate_bert(model, val_dataloader, device):
    model.eval()
    loss = []
    f1_scores = []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs in val_dataloader:
            inputs = inputs.to(device)
            preds = model(**inputs)
            loss.append(loss_fn(preds.detach(), inputs['labels']).cpu().item())
            f1_scores.append(f1_metric(preds.detach().cpu().numpy().argmax(axis=1), inputs['labels'].cpu()))

    return np.mean(loss), np.mean(f1_scores)


def finetune_bert():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = UtterancesBertDataset(
                                        data_folder='dataset/all_training_data/train/',
                                        label_path='dataset/all_training_data/train_labels.json',
                                        exception_files=['IS1002a', 'IS1005d', 'TS3012c'],
                                        tokenizer=tokenizer,
                                        max_len=128
    )
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = UtterancesBertDataset(
                                        data_folder='dataset/all_training_data/val/',
                                        label_path='dataset/all_training_data/val_labels.json',
                                        exception_files=['IS1002a', 'IS1005d', 'TS3012c'],
                                        tokenizer=tokenizer,
                                        max_len=128
    )
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model_name = 'bert-base-uncased'
    model = Bert(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # args = TrainingArguments(
    #     f"bert-finetune-pool-mlp",
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=4,
    #     num_train_epochs=5,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     metric_for_best_model='f1'
    # )
    # trainer = Trainer(
    #     model=bert,
    #     args=args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     tokenizer=tokenizer,
    #     compute_metrics=f1_metric
    # )
    #
    # trainer.train()
    lr = 5e-5
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 1
    eval_steps = 1

    run = wandb.init(
        project='default bert finetune',
        config={
            'lr': lr,
            'epochs': num_epochs
        }
    )

    # for epoch in range(num_epochs):
    epoch = 1
    model.train()
    batch_train_loss = []
    batch_train_f1 = []
    for i, inputs in tqdm.tqdm(enumerate(train_dataloader)):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        batch_train_f1.append(f1_metric(outputs.detach().cpu().numpy().argmax(axis=1), inputs['labels'].cpu()))
        loss = loss_fn(outputs, inputs['labels'])
        batch_train_loss.append(loss.detach().cpu())
        loss.backward()
        optimizer.step()

    if (i + 1) % eval_steps == 0:
        val_loss, val_f1 = evaluate_bert(model, val_dataloader, device)
        train_loss = np.mean(batch_train_loss)
        train_f1 = np.mean(batch_train_f1)

        metrics = {"train_loss": train_loss, "train_f1": train_f1,
                   "val_loss": val_loss, "val_f1": val_f1}
        print(metrics.items())
        run.log(metrics)

        torch.save(model.state_dict(), f"bert_finetuned_epoch{epoch}.pth")

        model.train()
        batch_train_loss = []
        batch_train_f1 = []

    scheduler.step()


if __name__ == '__main__':
    finetune_bert()





