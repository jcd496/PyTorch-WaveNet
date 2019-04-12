import torch
from torch.utils.data import DataLoader
from data import LJDataset, collate_fn
from model import WaveNet
import argparse

parser = argparse.ArgumentParser(description='WaveNet', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/scratch/jcd496/LJdata/processed', help='data root directory')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--num_workers', type=int, default=5, help='number of workers')
parser.add_argument('--blocks', type=int, default=4, help='number of workers')
parser.add_argument('--layers_per_block', type=int, default=5, help='number of workers')
args = parser.parse_args()

train_data = LJDataset(args.data_path, True, 0.1)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

model = WaveNet(args.blocks, args.layers_per_block, 1, 1, 1)
test=None
for idx, (x, c) in enumerate(train_loader):
    if idx==0:
        print(x)
        test = model(x)
 

print(test)
