####Data Handling reproduced from the work of Sungwon Kim et al as audio processing is out of the scope of this project
###Original work may be found on the FloWaveNet github
###https://github.com/ksw0306/FloWaveNet


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import transforms 

from utils import encode_mu_law, decode_mu_law
from scipy.io import wavfile

max_time_steps = 16000
hop_length = 256
receptive_field = 0
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
    
    x_batch = torch.tensor(x_batch_MuLaw, dtype=torch.float32).transpose(1,2).contiguous()##

    input_length = x_batch.shape[2]
    target_length = input_length - receptive_field
    target = x_batch[:,:,-target_length:]
    target = target.clone().detach().long()
    
    x_batch = one_hot_encode(x_batch)
    x_batch = torch.tensor(x_batch, dtype=torch.float32, requires_grad=False)#changed to False to support multiprocessing
    x_batch = x_batch.transpose(1,2)
    x_batch = x_batch[:,:,:-1]##
    return x_batch, target   ##c_batch

def data_generation(data, framerate, seq_size, mu, gen_mode=None):
    """
    Description : data generation to loading data
    """
    if gen_mode == 'sin':
        t = np.linspace(0, 5, framerate*5)
        data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
    div = max(data.max(), abs(data.min()))
    data = data/div
    while True:
        start = np.random.randint(0, data.shape[0]-seq_size)
        ys = data[start:start+seq_size]
        ys = encode_mu_law(ys, mu)
        yield torch.tensor(ys[:seq_size])


def data_generation_sample(data, framerate, seq_size, mu, gen_mode=None):
    """
    Description : sample data generation to loading data
    """
    if gen_mode == 'sin':
        t = np.linspace(0, 5, framerate*5)
        data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
    div = max(data.max(), abs(data.min()))
    data = data/div
    start = 0
    ys = data[start:start+seq_size]
    ys = encode_mu_law(ys, mu)
    return torch.tensor(ys[:seq_size])


def load_wav(file_nm):
    """
    Description : load wav file
    """
    fs, data = wavfile.read(os.getcwd()+'/data/'+file_nm)
    return  fs, data

def generate_slow(x, models, dilation_depth, n_repeat, n=100):
    """
    Description : module for generation core
    """
    dilations = [2**i for i in range(dilation_depth)] * n_repeat
    res = list(x.numpy())
    for i in range(n):
        x = torch.tensor(res[-sum(dilations)-1:])
        x = x.view(1, -1)
        x = one_hot_encode(x)
        x = torch.tensor(x, dtype=torch.float32)
        x = x.transpose(1, 2)
        y = models(x)
        y.squeeze()
        res.append(y.argmax(1).numpy()[-1])
    return res

def generation(mu, model):
    """
    Description : module for generation
    """
    fs, data = load_wav('parametric-2.wav')
    initial_data = data_generation_sample(data, fs, mu=mu, seq_size=3000)
    gen_rst = generate_slow(initial_data[0:3000], model, dilation_depth=10,\
            n_repeat=2, n=5000)
    gen_wav = np.array(gen_rst)
    gen_wav = decode_mu_law(gen_wav, 128)
    np.save("wav.npy", gen_wav)
