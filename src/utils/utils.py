import json
from pathlib import Path
import os
import numpy as np


def get_file_list(path, exts):
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            file_path = os.path.join(maindir, filename)
            ext = os.path.splitext(file_path)[1]
            if ext not in exts:
                file_names.append(file_path)
    return file_names


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def write_json(path, d):
    with open(path, "w") as file:
        file.write(json.dumps(d, indent=2))


def log_results(dest_path, model_desc, accuracy, f1_score):
    with open(dest_path, 'a') as f:
        f.write(f'{model_desc} has on val set: accuracy={accuracy} and f1_score={f1_score}\n')


def get_utterances_from_files(paths):
    file_names = [p.split('/')[-1].split('.')[0] for p in paths]

    utterances = []

    for name, path in zip(file_names, paths):
        transcription = load_json(path)
        for utt in transcription:
            utterances.append(utt["speaker"] + ": " + utt["text"])

    return utterances


def get_labels(labels_path, file_paths):
    file_names = [p.split('/')[-1].split('.')[0] for p in file_paths]

    labels = []
    labels_json = load_json(labels_path)

    for name in file_names:
        labels += labels_json[name]

    return labels


def to_lower_case(sentence):
    return sentence.lower()