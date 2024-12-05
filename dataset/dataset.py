from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
import numpy as np
from transformers import LLamaTokenizer
from sklearn.model_selection import train_test_split


class dataset(Dataset):
    DATA_TYPE = ['train', 'val', 'test']
    def __init__(self, root, dset_name, mode, label_id, 
                 model_name_or_path,
                 discrete=True, slicenum=100, truncate=3,
                 r=0.2):
        assert mode in self.DATA_TYPE, f'{mode} not in {self.DATA_TYPE}'
        # path = join(root, dset_name)
        path = root
        data_t = np.load(join(path, f'{dset_name}.npz'))
        label = data_t.files
        if 'fMRI' in dset_name:
            data_t = self.mean(data_t)
        label_id = label_id[dset_name]
        data = []
        labels = []
        for l in label:
            d = np.split(data_t[l], data_t[l].shape[0], axis=0)
            data.extend(d)
            labels.extend([label_id[l]] * data_t[l].shape[0])
    
        data = np.array(data).squeeze(1)
        labels = np.array(labels)

        data = self._l2_normalize_np_array(data)
        
        if discrete:
            data = self._discrete(data, slicenum)
        else:
            data = np.round(data, decimals=truncate)
        train, val, test = self.split(np.concatenate([data, labels[:, None]], axis=1))

        data = {
            'train': train,
            'test': test,
            'val': val,
        }
        self.labels = data[mode][:, -1]
        self.neg_pairs()
        self.data = self._str(data[mode][:, :-1])
        self.tokenizer = self.get_tokenizer(model_name_or_path)


    @staticmethod
    def _discrete(data, slicenum=100, eps=1e-18):
        # data (N, D) or (N, C, T)
        if len(data.shape) == 2:
            datamin = np.min(data, axis=0)
            datamax = np.max(data, axis=0)
            datamax = datamax + eps

            slices = np.linspace(datamin, datamax, slicenum)
            data = np.array([np.digitize(data[:, i], slices[:, i]) for i in range(data.shape[1])])
            data = data.astype(int)
            data = data.T
        elif len(data.shape) == 3:
            datamin = np.min(data, axis=(1, 2))
            datamax = np.max(data, axis=(1, 2))
            datamax = datamax + eps

            slices = np.linspace(datamin, datamax, slicenum)
            data = np.digitize(data, slices)
            data = data.astype(int)
        else:
            raise NotImplementedError(f'len(data.shape) !=2 && len(data.shape)!=3 Not Implement!')
        return data

    @staticmethod
    def _str(data):
        data_t = data.astype(str).tolist()
        data_t = [' '.join(d) for d in data_t]
        return data_t

    @staticmethod
    def _l2_normalize_np_array(np_array, eps=1e-5):
        """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
        return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

    def get_tokenizer(self, model_name_or_path: str):
        tokenizer = LLamaTokenizer.from_pretrained(
            model_name_or_path, padding_side='right', add_eos_token = True)
        if getattr(tokenizer, "pad_token_id") is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # output = self.tokenizer(self.data[idx])
        # token = output['input_ids']
        return self.data[idx], self.labels[idx]
    
    def neg_pairs(self, n=0.5, r=0.2):
        num = len(self.data) * n
        m_data = self.mask_(self.data)
        n_data = len(self.data)
        indices = np.random.choice(num, size=n, replace=(num > n_data))
        expand_data = m_data[indices]
        expand_label = self.labels[indices] * (1 - r)
        self.data = np.concatenate([self.data, expand_data], axis=0)
        self.labels = np.concatenate([self.labels, expand_label], axis=0)

    @staticmethod
    def mask_(data, r=0.2):
        mask = np.random.rand(*data.shape)
        mask = mask < r
        temp = np.copy(data)
        temp[mask] = 0
        return temp

    @staticmethod
    def split(data, val_=0.15, test_=0.15):
        d_train, d_valtest = train_test_split(data, test_size=(val_ + test_), random_state=2024)
        d_val, d_test = train_test_split(d_valtest, test_size=test_ / (val_ + test_), random_state=2024)
        return d_train, d_val, d_test

    def mean(self, rawdata):
        newdict = {}
        for key in rawdata.keys():
            newdata1 = []
            newdata2 = []
            for sample in rawdata[key]:
                newmean = np.mean(sample, axis=0)
                newvar = np.var(sample, axis=0)
                newdata1.append(newmean)
                newdata2.append(newvar)
            newdict[key] = np.concatenate((newdata1, newdata2), axis=1)
        return newdict

    def collate_fn(self, batch):
        tokens, labels = zip(*batch)
        input_dict = self.tokenizer(tokens, padding="longest", return_tensors="pt", truncation=True, max_length=None)
        input_dict['labels'] = torch.tensor(labels, dtype=torch.long)
        return input_dict


def build_dataset(args, mode):
    data_config = {
        'root': args.data_root,
        'dset_name': args.dset_name,
        'mode': mode,
        'label_id': args.label_id,
        'model_name_or_path': args.model_name_or_path,
        'discrete': args.discrete,
        'slicenum': args.slicenum,
        'truncate': args.truncate,
        'r': args.ratio,
    }

    if not args.discrete and args.slicenum != -1:
        raise RuntimeWarning('If discrete not True, slicenum will be ignored.')

    return dataset(**data_config)


if __name__ == '__main__':
    class opt:
        def __init__(self) -> None:
            self.discrete = False
            self.slicenum = 100
            self.truncate = 3
    data = dataset('../data', 'ADNI', 'train', 
                   label_id={'ADNI': {'NC': 0, 'AD': 1, 'MCI': 2, 'MCIn': 3, 'MCIp': 4}},
                   model_name_or_path='../Llama-2-7b-hf', 
                   discrete=True, slicenum=100, truncate=3)
    loader = DataLoader(data, batch_size=5, shuffle=True, collate_fn=data.collate_fn)
    for i in loader:
        print(i)
        break