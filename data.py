####Data Handling reproduced from the work of Sungwon Kim et al as audio processing is out of the scope of this project
###Original work may be found on the FloWaveNet github
###https://github.com/ksw0306/FloWaveNet


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import transforms 
max_time_steps = 16000
hop_length = 256

class LJDataset(Dataset):
    def __init__(self, root, train=True, test_size=0.05):
        self.root_dir = root
        self.train = train
        self.test_size = test_size
        self.paths = [self.get_files(0), self.get_files(1)]
    def __len__(self):
        return len(self.paths[0])
    def __getitem__(self, idx):
        wav = np.load(self.paths[0][idx])
        mel = np.load(self.paths[1][idx])
        return wav, mel
    def interest_indices(self,paths):
        test_num_samples = int(self.test_size * len(paths))
        train_indices, test_indices = range(0, len(paths) - test_num_samples), \
                range(len(paths) - test_num_samples, len(paths))
        return train_indices if self.train else test_indices

    def get_files(self, col):
        temp_path = os.path.join(self.root_dir, "train.txt")
        with open(temp_path, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        assert len(l) == 4
        self.lengths = list( map(lambda l: int(l.decode("utf-8").split("|")[2]), lines))
        paths = list(map(lambda l: l.decode("utf-8").split("|")[col], lines))
        paths = list(map(lambda f: os.path.join(self.root_dir, f), paths))

        indices = self.interest_indices(paths)
        paths = list(np.array(paths)[indices])
        self.lengthss = list(np.array(self.lengths)[indices])
        self.lengths = list(map(int, self.lengths))
        return paths

def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0,0)], mode="constant", constant_values=0)
    return x

def one_hot_encode(targets, num_classes=256):
    #takes targets as numpy array
    one_hots = np.eye(num_classes)[targets]
    return one_hots

def collate_fn(batch):

    local_conditioning = len(batch[0]) >= 2

    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c = batch[idx]
            
            assert len(x) %len(c) == 0 and len(x) // len(c) == hop_length
            max_steps = max_time_steps - max_time_steps % hop_length

            if len(x) > max_steps:
                max_time_frames = max_steps // hop_length
                s = np.random.randint(0, len(c) - max_time_frames)
                ts = s*hop_length
                x = x[ts:ts + hop_length*max_time_frames]
                c = c[s:s + max_time_frames]
                assert len(x) % len(c) == 0 and len(x) // len(c) == hop_length
            new_batch.append((x,c))
        batch=new_batch
    else:
        pass

    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    x_batch = np.array([_pad_2d(x[0].reshape(-1,1), max_input_len) for x in batch], dtype = np.float32)
    assert len(x_batch.shape) == 3

    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype = np.float32)
        assert len(c_batch.shape) == 3
        c_batch = torch.tensor(c_batch).transpose(1,2).contiguous()
        del max_len
    else:
        c_batch = None
    MuTransform = transforms.Compose([ transforms.MuLawEncoding() ])
    x_batch_MuLaw = MuTransform(x_batch)
    x_batch = torch.tensor(x_batch_MuLaw, dtype=torch.float32).transpose(1,2).contiguous()##
    x_batch = x_batch[:,:,:-1]##
    #x_batch = x_batch[:,:,:-1024]##for prediciting residual field size target
    #target = x_batch[:,:,-1024:].clone().detach().long()##for predicting residual field size target
    target = x_batch[:,:,-1:].clone().detach().long()##
    #target = one_hot_encode(target) ##
    #target = torch.tensor(target, dtype=torch.int32) ##
    return x_batch, target   ##c_batch
