import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from model import WaveNet
import data as D 
#import LJDataset, collate_fn
from time import monotonic
import argparse
import sys


parser = argparse.ArgumentParser(description='WaveNet', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/scratch/jcd496/LJdata/processed', help='data root directory')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of workers')
parser.add_argument('--blocks', type=int, default=1, help='number of blocks of residual layers')
parser.add_argument('--layers_per_block', type=int, default=10, help='residual layers per block')
parser.add_argument('--use_cuda', type=bool, default=False, help='offload to gpu')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--mu', type=int, default=128, help='number of epochs')
parser.add_argument('--dataset', type=str, default='ljdataset', help="The dataset to use. Can be 'ljdataset' or 'gluon'")
args = parser.parse_args()

D.receptive_field = (sum([2**l for l in range(args.layers_per_block)]))*args.blocks + 1
print(D.receptive_field)
train_data = D.LJDataset(args.data_path, True, 0.1)
print("Number of training inputs:", len(train_data))
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=D.collate_fn, num_workers=args.workers)
device = torch.device('cpu')
if args.use_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
class timer():
    def __init__(self):
        self.count=0
        self.sum=0.0
        self.avg=0.0
    def update(self, time):
        self.sum+=time
        self.count+=1
    def average(self):
        return self.sum/self.count

def train(model, optimizer, criterion):
    print("LJdataset training.")
    epoch_time = timer()
    losses = []
    for epoch in range(args.epochs):
        running_loss = 0.0
        epoch_s = monotonic()
        for idx, (x, target) in enumerate(train_loader):
            target = target.view(-1) 
            x, target = x.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            output = model(x)
            print(output.shape, target.shape)            
            loss = criterion(output.squeeze(), target.squeeze())
            
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
        epoch_f = monotonic()
        epoch_time.update((epoch_f - epoch_s))
        losses.append(running_loss)
        print("loss {:.3f} time {:.3f}".format(running_loss,epoch_time.average()))
    return losses


def gluon_train(model, loss_fn, optimizer, mu, seq_size, epochs, batch_size):
    """
    Description : module for running train
    """

    print("Gluon training.")
    fs, data = D.load_wav('parametric-2.wav')
    g = D.data_generation(data, fs, mu=mu, seq_size=seq_size)
    loss_save = []
    best_loss = sys.maxsize
    predictions = None
    for epoch in range(epochs):
        loss_tot = 0.0
        for i in range(batch_size):
            batch = next(g)
            batch = batch.view(1, -1)
            x = batch[:,:-1]
            x = D.one_hot_encode(x)
            x = torch.tensor(x, dtype=torch.float32)
            x = x.transpose(1, 2)
            optimizer.zero_grad()
            logits = model(x)
            predictions = logits.cpu().detach()
            sz = logits.shape[0]
            loss = loss_fn(logits, batch[0,-sz:])
            loss.backward(retain_graph=True)
            loss_tot += loss.item()
            optimizer.step()
        loss_save.append(loss_tot)

        #save the best model
        current_loss = loss_tot
        if best_loss > current_loss:
            print('epoch {}, loss {}'.format(epoch, loss_tot))
            #self.save_model(epoch, current_loss)
            best_loss = current_loss
    print("Best loss:", best_loss)
    return loss_save, predictions

        
if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device('cuda')
    print("Device:", device)
    model = WaveNet(args.blocks, args.layers_per_block, 24, 256)

    #optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Rprop(model.parameters()) 
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)


    if (args.dataset == 'gluon'):
        ### Gluon train
        mu = 128
        seq_size = 20000
        epochs = args.epochs
        batch_size = 64
        losses, predictions = gluon_train(model, criterion, optimizer, mu, seq_size, epochs, batch_size)
        D.generation(mu, model)
    else:
         ### LJDataset
        losses = train(model, optimizer, criterion)
        print(losses)


