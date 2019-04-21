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
        self.paths = self.get_files(0)##
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        wav = np.load(self.paths[idx])
        return wav
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
    one_hots = []
    for target in targets:
        target = target.long().numpy().reshape(-1)
        one_hots.append(np.eye(num_classes)[target])
    return np.array(one_hots)

def collate_fn(batch):
    local_conditioning = len(batch[0]) >= 2

    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x = batch[idx]
            
            max_steps = max_time_steps - max_time_steps % hop_length

            if len(x) > max_steps:
                max_time_frames = max_steps // hop_length
                s = np.random.randint(0,  max_time_frames)
                ts = s*hop_length
                x = x[ts:ts + hop_length*max_time_frames]
            new_batch.append((x))
        batch=new_batch
    else:
        pass
    input_lengths = [len(x) for x in batch]
    max_input_len = max(input_lengths)
    MuTransform = transforms.Compose([ transforms.MuLawEncoding() ])
    x_batch_MuLaw = np.array([MuTransform(_pad_2d(x.reshape(-1,1), max_input_len)) for x in batch], dtype = np.float32)
    assert len(x_batch_MuLaw.shape) == 3
    x_batch = torch.tensor(x_batch_MuLaw, dtype=torch.float32, requires_grad=True).transpose(1,2).contiguous()##
    x_batch = x_batch[:,:,:-1]##
    target = x_batch[:,:,-1023:].clone().detach().long()##
    x_batch = one_hot_encode(x_batch)
    x_batch = torch.tensor(x_batch, dtype=torch.float32, requires_grad=True)
    x_batch = x_batch.transpose(1,2)
    #x_batch = x_batch[:,:,:-1024]##for prediciting residual field size target
    #target = x_batch[:,:,-1024:].clone().detach().long()##for predicting residual field size target
    
    #target = one_hot_encode(target) ##
    #target = torch.tensor(target, dtype=torch.int32) ##
    return x_batch, target   ##c_batch
