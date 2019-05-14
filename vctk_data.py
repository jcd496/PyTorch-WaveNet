import os
import numpy as np
import torch
from torch.utils.data import Dataset
import transforms 
import torch.nn.functional as F
import fnmatch

max_time_steps = 16000
hop_length = 256
receptive_field = 0
class VCTKDataset(Dataset):
    def __init__(self, root, train=True, test_size=0.05):
        self.root_dir = root
        self.train = train
        self.test_size = test_size
        self.paths, self.speakers = self.get_files()
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        wav = np.load(self.paths[idx])
        speaker = self.speakers[idx]
        return wav, speaker
    def get_files(self):
        files = []
        speakers = {}
        for root, dirnames, filenames in os.walk(self.root_dir):
            for filename in fnmatch.filter(filenames, '*.npy'):
                files.append(os.path.join(root, filename))
        speakers = []
        encoding = []
        speaker = files[0].split('_')[0]
        speakers.append(speaker)
        for speaker in files:
            speak = speaker.split('_')[0]
            if speak in speakers: encoding.append(speakers.index(speak))
            else:
                speakers.append(speak)
                encoding.append(speakers.index(speak))
        self.num_speakers = max(encoding)+1
        print("number of speakers", self.num_speakers)
        encoding = torch.tensor(encoding)
        encoding = one_hot_encode(encoding, num_classes = self.num_speakers).squeeze()
        return files, encoding
        
        
def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0,0)], mode="constant", constant_values=0)
    return x

def one_hot_encode(targets, num_classes=256):
    #takes targets as numpy array
    one_hots = []
    for target in targets:
        target = target.long().cpu().numpy().reshape(-1)
        one_hots.append(np.eye(num_classes)[target])
    return np.array(one_hots)


def collate_fn(batch):
    local_conditioning = len(batch[0]) >= 2
    '''
    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, s = batch[idx]
            max_steps = max_time_steps - max_time_steps % hop_length

            if len(x) > max_steps:
                max_time_frames = max_steps // hop_length
                t = np.random.randint(0, max_time_frames)
                ts = t*hop_length
                x = x[ts:ts+hop_length*max_time_frames]
                new_batch.append((x, s))
        batch = new_batch
    else: pass
    ''' 
    input_lengths = [len(x) for x, s in batch]
    max_input_len = max(input_lengths)
    speakers = torch.tensor([s for x, s in batch], dtype=torch.float)
    MuTransform = transforms.Compose([ transforms.MuLawEncoding() ])
    x_batch_MuLaw = np.array([MuTransform(_pad_2d(x.reshape(-1,1), max_input_len)) for x, s in batch], dtype = np.float32)
    assert len(x_batch_MuLaw.shape) == 3
    x_batch = torch.tensor(x_batch_MuLaw, dtype=torch.float32).transpose(1,2).contiguous()##
    x_batch = F.pad(x_batch, pad=(1,0)) #left pad with single 0

    input_length = x_batch.shape[2]
    target_length = input_length - receptive_field
    target = x_batch[:,:,-target_length:].abs()
    #target = x_batch[:,:,-(target_length-1):] #to shift output right by single element
    target = target.clone().detach().long()
    
    x_batch = one_hot_encode(x_batch)
    x_batch = torch.tensor(x_batch, dtype=torch.float32, requires_grad=False)#changed to False to support multiprocessing
    x_batch = x_batch.transpose(1,2)
    x_batch = x_batch[:,:,:-1]##
    return x_batch, speakers, target   ##c_batch
