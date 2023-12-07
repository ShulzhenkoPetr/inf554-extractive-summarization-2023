import json
from pathlib import Path
import os
import shutil
import argparse
import numpy as np

from utils import get_file_list, load_json, write_json


def train_val_split(args):
    data_path = Path(args.data_path)

    all_jsons = get_file_list(data_path, ['.txt'])
    rng = np.random.default_rng(12345)

    val_indices = rng.choice(len(all_jsons),
                             size=int(0.2 * len(all_jsons)),
                             replace=False, shuffle=False)

    train_indices = [idx for idx in range(len(all_jsons))
                     if idx not in val_indices]

    val_files = [all_jsons[idx].split('/')[-1].split('.')[0]
                 for idx in val_indices]
    train_files = [all_jsons[idx].split('/')[-1].split('.')[0]
                   for idx in train_indices]

    dest_path = Path('/'.join(args.data_path.split('/')[:-1]) + '/')

    if not os.path.exists(dest_path / 'val'):
        os.makedirs(dest_path / 'val')
    if not os.path.exists(dest_path / 'train'):
        os.makedirs(dest_path / 'train')

    for idx, file in zip(val_indices, val_files):
        shutil.copyfile(all_jsons[idx],
                        dest_path / 'val' / f'{file}.json')
        shutil.copyfile(all_jsons[idx].replace('json', 'txt'),
                        dest_path / 'val' / f'{file}.txt')

    for idx, file in zip(train_indices, train_files):
        shutil.copyfile(all_jsons[idx],
                        dest_path / 'train' / f'{file}.json')
        shutil.copyfile(all_jsons[idx].replace('json', 'txt'),
                        dest_path / 'train' / f'{file}.txt')

    labels = load_json('dataset/training_labels.json')
    val_dict = {}
    train_dict = {}
    for key, value in labels.items():
        if key in val_files:
            val_dict[key] = value
        else:
            train_dict[key] = value

    write_json('dataset/all_training_data/val_labels.json', val_dict)
    write_json('dataset/all_training_data/train_labels.json', train_dict)

    return train_files, val_files


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default='dataset/all_training_data/')
    parser.add_argument('--ratio', type=float, default=0.2)

    args = parser.parse_args()

    _, _ = train_val_split(args)


if __name__ == '__main__':
    run()