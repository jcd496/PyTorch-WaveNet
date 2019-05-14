import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import WaveNet
import data as D 
from transforms import MuLawExpanding, to_wav
from time import monotonic
import time
import argparse
import sys
import numpy as np
from bach_data import BachDataset
import transforms

parser = argparse.ArgumentParser(description='WaveNet', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/scratch/jcd496/LJdata/processed', help='data root directory')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of workers')
parser.add_argument('--blocks', type=int, default=1, help='number of blocks of residual layers')
parser.add_argument('--layers_per_block', type=int, default=10, help='residual layers per block')
parser.add_argument('--use_cuda', type=bool, default=False, help='offload to gpu')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--mu', type=int, default=128, help='number of epochs')
parser.add_argument('--dataset', type=str, default='ljdataset', help="The dataset to use. Can be 'ljdataset', 'bach', 'ljtest', or 'bachtest'")
parser.add_argument('--save_path', type=str, default=None, help='path to save trained model')
parser.add_argument('--model_name', type=str, default=None, help='path name of model')
parser.add_argument('--load_path', type=str, default=None, help='path to load saved model')
parser.add_argument('--save_wav', type=str, default=None, help='path to save wav prediction')
parser.add_argument('--speakers', type=int, default=5, help='number of speakers used in global conditioning')
parser.add_argument('--test_ratio', type=float, default=0.99, help='ratio of data to use for testing. The rest is used for training.')
parser.add_argument('--gen_len', type=int, default=1, help='number of seconds of data to generate')
parser.add_argument('--stride', type=int, default=500, help='stride to use in bach or bachtest dataloader.')
parser.add_argument('--save_interval', type=int, default=None, help='Save model every <interval> epochs. Default 0.1 * epochs')
parser.add_argument('--residual_channels', type=int, default=32, help='number of residual channels to use')
parser.add_argument('--dilation_channels', type=int, default=32, help='number of dilation channels to use')
parser.add_argument('--skip_channels', type=int, default=1024, help='number of skip channels to use')
parser.add_argument('--end_channels', type=int, default=512, help='number of end channels to use')


args = parser.parse_args()

if not args.save_interval:
    args.save_interval = int(0.1 * args.epochs)
if args.save_path:
    if not args.model_name:
        print("Error: --save_path specified but no --model_name specified.")
        quit()
    if args.save_path[-1] == '/':
        args.save_path = args.save_path[:-1]

D.receptive_field = (sum([2**l for l in range(args.layers_per_block)]))*args.blocks + 1
test_data = None
train_data = None
test_loader = None
train_loader = None

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

def train(model, optimizer, dataloader, criterion, epochs, evaluate = True):
    model.train()
    epoch_time = timer()
    losses = []
    predictions = None

    for epoch in range(epochs):
        running_loss = 0.0
        epoch_s = monotonic()
        for idx, (x, target) in enumerate(dataloader):
            target = target.view(-1) 
            x, target = x.to(device), target.to(device)
            output = model(x)
            loss = criterion(output.squeeze(), target.squeeze())
            optimizer.zero_grad()
            loss.backward()
            running_loss+=loss.item()
            optimizer.step()
            predictions = output.cpu().detach() 
            print((idx + 1) * train_loader.batch_size)
        # indent from for loop to here
        epoch_f = monotonic()
        epoch_time.update((epoch_f - epoch_s))
        losses.append(running_loss)
        print("epoch {} avg_loss {:.3f} time {:.3f}".format(epoch, running_loss/len(dataloader),epoch_time.average()))
    
        if args.save_path:
            if (epoch + 1) % args.save_interval == 0:
                model.to(torch.device('cpu'))
                state = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}
                t = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                pth = args.save_path + '/' + args.model_name + '_' + t + '.pt'
                torch.save(state, pth)
                model.to(device)
    return losses, predictions


def VCTK_train(model, optimizer, criterion):
    print("VCTK dataset training.")
    epoch_time = timer()
    losses = []
    predictions = None
    train_data = D.VCTKDataset(args.data_path, True, 0.1)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=D.vctk_collate_fn, num_workers=args.workers)
    print("Number of training inputs:", len(train_data))
    for idx, (x, gc, target) in enumerate(train_loader):
        break
    target = target.view(-1) 
    x, gc, target = x.to(device), gc.to(device), target.to(device)
    for epoch in range(args.epochs):
        running_loss = 0.0
        epoch_s = monotonic()
        #for idx, (x, gc, target) in enumerate(train_loader):
        #    target = target.view(-1) 
        #    x, gc, target = x.to(device), gc.to(device), target.to(device)

        optimizer.zero_grad()
        #gc = torch.tensor([[1,0,0,0,0] for i in range(5)]).float().to(device)        
        #gc = torch.tensor([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]).float().to(device)        
        output = model(x, gc)
        loss = criterion(output.squeeze(), target.squeeze())
        predictions = output 
        loss.backward()
        
        optimizer.step()
        running_loss+=loss.item()
        epoch_f = monotonic()
        epoch_time.update((epoch_f - epoch_s))
        losses.append(running_loss)
        print("epoch {} loss {:.3f} time {:.3f}".format(epoch, running_loss, epoch_time.average()))
    predictions = predictions.cpu().detach()    
    if args.save_wav:
        values, predictions = torch.topk(predictions, 1, dim=1)
        expander = MuLawExpanding()
        predictions = expander(predictions.squeeze().numpy())
        predictions = predictions.astype(np.float32)
        to_wav(args.save_wav, predictions, sample_rate=48000)

    return losses, predictions

def LJ_train(model, optimizer, criterion, evaluate = True):
    model.train()
    print("LJdataset training.")
    epoch_time = timer()
    losses = []
    predictions = None
    for epoch in range(args.epochs):
        running_loss = 0.0
        epoch_s = monotonic()
        for idx, (x, target) in enumerate(train_loader):
            target = target.view(-1) 
            x, target = x.to(device), target.to(device)
            output = model(x)
            loss = criterion(output.squeeze(), target.squeeze())
            optimizer.zero_grad()
            predictions = output.cpu().detach() 
            loss.backward()
            running_loss+=loss.item()
            optimizer.step()
            #print((idx + 1) * 25)
        ##indent from for loop to here
        epoch_f = monotonic()
        epoch_time.update((epoch_f - epoch_s))
        losses.append(running_loss)
        print("epoch {} avg_loss {:.3f} time {:.3f}".format(epoch, running_loss/len(train_loader),epoch_time.average()))

        if args.save_path:
            if (epoch + 1) % args.save_interval == 0:
                model.to(torch.device('cpu'))
                state = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}
                t = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                pth = args.save_path + '/' + args.model_name + '_' + t + '.pt'
                torch.save(state, pth)
                model.to(device)
                #lss, acc = LJ_evaluate(model, optimizer, criterion)
                #print("epoch {} loss {:.3f} acc {:.3f}".format(epoch, lss, acc))

    return losses, predictions


def evaluate(model, optimizer, criterion, test_loader):
    print("Device:", device)
    model.eval()
    running_loss = 0
    correct = 0

    for idx, (x, target) in enumerate(test_loader):
        #print((idx + 1) * args.batch_size)
        target = target.view(-1)
        x, target = x.to(device), target.to(device)
        output = model(x)
        loss = criterion(output.squeeze(), target.squeeze())
        running_loss += loss.item()
        predictions = torch.max(output.cpu().detach(), 1)[1].view(-1)
        #print("target length", model.target_length)
        #print("predictions shape b4 view", predictions.shape)
        predictions  = predictions.view(-1)
        #print(predictions[14748:14848])
        #print(target[14748:14848])
        correct_pred = torch.eq(target.cpu(), predictions)
        correct += torch.sum(correct_pred).item()

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / (len(test_data) * model.target_length)
    model.train()
    return avg_loss, accuracy

def LJ_evaluate(model, optimizer, criterion):
    print("Device:", device)
    model.eval()
    running_loss = 0
    correct = 0

    for idx, (x, target) in enumerate(test_loader):
        #print((idx + 1) * args.batch_size)
        target = target.view(-1)
        x, target = x.to(device), target.to(device)
        output = model(x)
        loss = criterion(output.squeeze(), target.squeeze())
        running_loss += loss.item()
        predictions = torch.max(output.cpu().detach(), 1)[1].view(-1)
        #print("target length", model.target_length)
        #print("predictions shape b4 view", predictions.shape)
        predictions  = predictions.view(-1)
        #print(predictions[14748:14848])
        #print(target[14748:14848])
        correct_pred = torch.eq(target.cpu(), predictions)
        correct += torch.sum(correct_pred).item()

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / (len(test_data) * model.target_length)
    model.train()
    return avg_loss, accuracy


def predict(model):
    model.eval()
    x, target = next(iter(train_loader))
    x, target = x.to(device), target.to(device)
    r = model(x).cpu().detach()
    model.train()
    return r

        
if __name__ == '__main__':
    print("batch_size", args.batch_size)
    print("blocks", args.blocks)
    print("layers_per_block", args.layers_per_block)
    print("use_cuda", args.use_cuda)
    print("epochs", args.epochs)
    print("dataset", args.dataset)
    print("save_path", args.save_path)
    print("model_name", args.model_name)
    print("load_path", args.load_path)
    print("save_wav", args.save_wav)
    print("test_ratio", args.test_ratio)
    print("gen_len", args.gen_len)
    print("stride", args.stride)
    print("save_interval", args.save_interval)
    print("residual_channels", args.residual_channels)
    print("dilation_channels", args.dilation_channels)
    print("skip_channels", args.skip_channels)
    print("end_channels", args.end_channels)

    device = torch.device('cpu')
    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device('cuda')
    print("Device:", device)

    GLOBAL_CONDITIONING = False
    if (args.dataset == 'VCTK'):
        GLOBAL_CONDITIONING = True

    model = WaveNet(args.blocks, args.layers_per_block, GLOBAL_CONDITIONING, args.speakers, \
                    output_channels=256, residual_channels=args.residual_channels, dilation_channels=args.dilation_channels, \
                    skip_channels=args.skip_channels, end_channels=args.end_channels)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    if args.load_path:
        state = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(state['model'])
        model.to(device)

        #temp bug fix 
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        optimizer.load_state_dict(state['optimizer'])


    losses = None
    predictions = None 
    mu = 256
    if args.dataset == 'ljtest':
        D.target_length = 16
        test_data = D.LJDataset(args.data_path, False, args.test_ratio)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=D.collate_fn, num_workers=args.workers)
        lss, acc = LJ_evaluate(model, optimizer, criterion)
        print("loss {:.3f} acc {:.3f}".format(lss, acc))
        quit()
    elif args.dataset == 'bachtest':
        output_length = 16
        test_data = BachDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                              item_length=D.receptive_field + output_length - 1,
                              target_length=output_length,
                              file_location='train_samples/bach_chaconne',
                              test_stride=args.stride,
                              train=False)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        lss, acc = evaluate(model, optimizer, criterion, test_loader)
        print("loss {:.3f} acc {:.3f}".format(lss, acc))
        quit()
    elif args.dataset == 'ljdataset':
         ### LJDataset
        #losses, predictions = LJ_train(model, optimizer, criterion)
        D.target_length = 16
        train_data = D.LJDataset(args.data_path, True, args.test_ratio)
        test_data = D.LJDataset(args.data_path, False, args.test_ratio)
        print("Number of training inputs:", len(train_data))
        print("Number of test inputs:", len(test_data))
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=D.collate_fn, num_workers=args.workers)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=D.collate_fn, num_workers=args.workers)
        losses, predictions = LJ_train(model, optimizer, criterion, evaluate=True)
    
    elif args.dataset == 'VCTK':
        ### VCTKDataset
        losses, predictions = VCTK_train(model, optimizer, criterion)
        
    elif args.dataset == 'bach':
        output_length = 16
        train_data = BachDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                              item_length=D.receptive_field + output_length - 1,
                              target_length=output_length,
                              file_location='train_samples/bach_chaconne',
                              test_stride=args.stride)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        print("Number of training inputs:", len(train_data))
        print('Bach training')
        losses, predictions = train(model, optimizer, train_loader, criterion, args.epochs, False)
    else:
        print("Invalid dataset.")
        quit()


    if args.save_path:
        model.to(torch.device('cpu'))
        state = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}
        pth = args.save_path + '/' + args.model_name + '.pt'
        print(pth)
        torch.save(state, pth)

    torch.cuda.empty_cache()

    if args.save_path:
        save_wav = args.save_path + '/' + args.model_name
        D.generation(mu, model, device, filename = save_wav, seconds = args.gen_len, dataset = args.dataset)

        if args.dataset != 'bach':
            predictions = predict(model)
            values, predictions = torch.topk(predictions, 1, dim=1)
            expander = MuLawExpanding()
            predictions = expander(predictions.squeeze().numpy())
            predictions = predictions.astype(np.float32)
            pth = save_wav + '_output.wav'
            print(pth)
            to_wav(pth, predictions)

    if losses:
        print(losses)
