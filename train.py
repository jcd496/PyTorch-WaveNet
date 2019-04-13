import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from data import LJDataset, collate_fn
from model import WaveNet
import argparse

parser = argparse.ArgumentParser(description='WaveNet', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/scratch/jcd496/LJdata/processed', help='data root directory')
parser.add_argument('--batch_size', type=int, default=250, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
parser.add_argument('--blocks', type=int, default=4, help='number of blocks of residual layers')
parser.add_argument('--layers_per_block', type=int, default=5, help='residual layers per block')
parser.add_argument('--use_cuda', type=bool, default=False, help='offload to gpu')
args = parser.parse_args()

train_data = LJDataset(args.data_path, True, 0.1)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
device = torch.device('cpu')
if args.use_cuda and torch.cuda.is_available():
    device = torch.device('cuda')

def train(model, epochs, data_loader, optimizer, criterion):
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, (x, target) in enumerate(train_loader):
            target = target.view(-1)  
            x, target = x.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            running_loss+=loss
            print(running_loss)
        print("loss", running_loss)

        
if __name__ == '__main__':


    model = WaveNet(args.blocks, args.layers_per_block, 1, 1, 1,args.batch_size)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train(model, 10, train_loader, optimizer, criterion)


