import torch
from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample
import tqdm
import numpy as np

from utils.utils import get_file_list, get_utterances_from_files, get_labels, get_speaker_relation_utterance_from_files


class UtterancesDataset(Dataset):
    """
    Creates a dataset from only utterances and speakers
    :param data_folder = path to train/val/test folder with jsons and txts
    :param exception_files = files to ignore
    """

    def __init__(self, data_folder, label_path, exception_files, preproc=None):
        self.data_folder = data_folder
        self.label_path = label_path
        self.exception_files = exception_files

        folder_file_paths = get_file_list(data_folder, ['.txt'])
        self.file_paths = [p for p in folder_file_paths
                           if p.split('/')[-1].split('.')[0] not in exception_files]
        self.utterances = get_utterances_from_files(self.file_paths)
        self.labels = get_labels(label_path, self.file_paths)

        assert len(self.utterances) == len(self.labels)

        self.preproc = preproc

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        if self.preproc:
            return self.preproc(self.utterances[idx]), self.labels[idx]

        return self.utterances[idx], self.labels[idx]


class UtterancesBertDataset(Dataset):
    def __init__(self, data_folder,
                 label_path,
                 exception_files,
                 tokenizer,
                 max_len,
                 preproc=None):
        self.data_folder = data_folder
        self.label_path = label_path
        self.exception_files = exception_files

        folder_file_paths = get_file_list(data_folder, ['.txt'])
        self.file_paths = [p for p in folder_file_paths
                           if p.split('/')[-1].split('.')[0] not in exception_files]
        self.utterances = get_utterances_from_files(self.file_paths)
        self.labels = get_labels(label_path, self.file_paths)

        assert len(self.utterances) == len(self.labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preproc = preproc

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = self.utterances[idx]

        if self.preproc:
            utt = self.preproc(utt)

        enc = self.tokenizer(utt, max_length=self.max_len, return_tensors='pt',
                             padding='max_length', truncation=True)
        enc['input_ids'] = torch.reshape(enc['input_ids'], (-1,))
        enc['labels'] = self.labels[idx]

        return enc


def triplet_data(data_folder,
                 label_path,
                 exception_files,):
    """
    Function that creates list of triples of anchor, positive
    and negative sentences, where a sentence is a combination
    of speaker + discourse relation + utterance
    """

    folder_file_paths = get_file_list(data_folder, ['.txt'])
    file_paths = [p for p in folder_file_paths
                       if p.split('/')[-1].split('.')[0] not in exception_files]
    sentences = get_speaker_relation_utterance_from_files(file_paths)
    labels = get_labels(label_path, file_paths)
    assert len(sentences) == len(labels)

    # self.encoded_sentences = self.tokenizer(self.sentences, max_length=self.max_len, return_tensors='pt',
    #                                         padding='max_length', truncation=True)
    ones = []
    zeros = []
    for sen, label in zip(sentences, labels):
        if int(label) == 0:
            zeros.append(sen)
        else:
            ones.append(sen)

    positive_len = len(ones)
    # data = [{
    #     'tokenized_anchor': ones[np.random.randint(positive_len)],
    #     'tokenized_positive': ones[np.random.randint(positive_len)],
    #     'tokenized_negative': zeros[i]
    # } for i in range(len(zeros))]
    data = [InputExample(texts=[ones[np.random.randint(positive_len)],
                                ones[np.random.randint(positive_len)],
                                zeros[i]]
                         ) for i in range(len(zeros))]

    return data


def sentences_data(data_folder,
                   label_path,
                   exception_files):
    folder_file_paths = get_file_list(data_folder, ['.txt'])
    file_paths = [p for p in folder_file_paths
                  if p.split('/')[-1].split('.')[0] not in exception_files]
    sentences = get_speaker_relation_utterance_from_files(file_paths)
    labels = get_labels(label_path, file_paths)
    assert len(sentences) == len(labels)

    return sentences, labels
