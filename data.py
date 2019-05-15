
####Data Handling for LJDataset is reproduced and significantly modified from the work of Sungwon Kim et al 
###Original work may be found on the FloWaveNet github
###https://github.com/ksw0306/FloWaveNet


import os
import numpy as np
import fnmatch
import torch
from torch.utils.data import Dataset
import transforms 
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import wavfile

max_time_steps = 16000
hop_length = 256
receptive_field = 0
target_length = 0
class LJDataset(Dataset):
    def __init__(self, root, train=True, test_size=0.05):
        self.root_dir = root
        self.train = train
        self.test_size = test_size
        self.paths = self.get_files(0)##
    def __len__(self):
        #return 2000
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
        self.lengths = list(np.array(self.lengths)[indices])
        self.lengths = list(map(int, self.lengths))
        return paths

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
    input_length = receptive_field + target_length

    if len(batch[0]) >= 2:
        new_batch = []
        for idx in range(len(batch)):
            x = batch[idx]
            
            if len(x) > input_length:
                s = np.random.randint(0,  len(x) - input_length)
                x = x[s:s + input_length]
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
   
    target = x_batch[:,:,-target_length:]

    target = target.clone().detach().long()
    
    x_batch = one_hot_encode(x_batch)
    x_batch = torch.tensor(x_batch, dtype=torch.float32, requires_grad=False)#changed to False to support multiprocessing
    x_batch = x_batch.transpose(1,2)

    x_batch = x_batch[:,:,:-1]
    return x_batch, target 
         
def data_generation_sample(data, seq_size, start = 0, dataset = 'ljdataset'):
    """
    Description : sample data generation to loading data
    """
    ys = data[start:start+seq_size]
    if dataset == 'bach':
        ys = torch.tensor(ys[:seq_size])
    else:
        encoder = transforms.MuLawEncoding()
        ys = encoder(ys)
        ys = torch.tensor(ys[:seq_size])

    return ys


def load_wav(file_nm):
    """
    Description : load wav file
    """
    fs, data = wavfile.read(os.getcwd()+'/'+file_nm)
    return  fs, data


def generate_slow(x, model, device, dilation_depth, n_repeat, n=100, sample=False):
    """
    Description : module for generation core
    """
    model.eval()
    dilations = [2**i for i in range(dilation_depth)] * n_repeat
    res = list(x)
   
    for i in range(n):
        x = torch.tensor(res[-sum(dilations)-1:])
        x = x.view(1, -1)
        x = F.pad(x, pad=(1,0))
        x = one_hot_encode(x)
        x = torch.tensor(x, dtype=torch.float32)
        x = x.transpose(1, 2)
        x = x.to(device)
        y = model(x)
        if sample:
            dist = F.softmax(y, dim=1)
            dist = dist.cpu().detach()
            np_dist = dist.numpy()[0]
            r = np.random.choice(256, p=np_dist)
        else:
            y = y.cpu().detach()
            r = y.argmax(1).numpy()[-1]
        res.append(r)
    model.train()
    return res
def generation(model, device, filename = None, seconds = 1, dataset = 'ljdataset'):
    if not filename:
        filename = 'wav'
    print("Generating...")
    """
    Description : module for generation
    """
    if dataset == 'bach':
        data = np.load('train_samples/bach_chaconne/dataset.npz', mmap_mode='r')['arr_0'] # already mu law encoded
        fs = 16000
    else:
        fs = 22050
        data = np.load("/scratch/jcd496/LJdata/processed/ljspeech-audio-00001.npy")
    
    s = fs
    
    num_to_gen = seconds * fs
    initial_data = data_generation_sample(data, fs, seq_size=s, start = (6 * 60 + 1) * fs if dataset == 'bach' else fs, dataset = dataset)
 
    gen_rst = generate_slow(initial_data[0:4000], model, device, dilation_depth=10,\
            n_repeat=model.num_blocks, n=num_to_gen, sample=False)
    gen_wav = np.array(gen_rst)
    plt.plot(gen_wav, ',')
    pth = filename + ".jpg"
    print(pth)
    plt.savefig(pth)
    expander = transforms.MuLawExpanding()
    gen_wav2 = expander(gen_wav).astype(np.float32)
    pth = filename + ".wav"
    print(pth)
    transforms.to_wav(pth, gen_wav2, sample_rate=fs)
    print("Generation complete.")
    


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
        
        

def vctk_collate_fn(batch):
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


