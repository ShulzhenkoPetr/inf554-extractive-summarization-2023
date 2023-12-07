from torch.utils.data import Dataset

from utils.utils import get_file_list, get_utterances_from_files, get_labels


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
                             padding=True, truncation=True)

        return enc, self.labels[idx]

