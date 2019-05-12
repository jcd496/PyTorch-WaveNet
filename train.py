import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from model import WaveNet
import data as D 
from transforms import MuLawExpanding, to_wav
from time import monotonic
import time
import argparse
import sys
import numpy as np
import gc

from audio_data import WavenetDataset

import transforms

import audio_data

parser = argparse.ArgumentParser(description='WaveNet', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/scratch/jcd496/LJdata/processed', help='data root directory')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of workers')
parser.add_argument('--blocks', type=int, default=1, help='number of blocks of residual layers')
parser.add_argument('--layers_per_block', type=int, default=10, help='residual layers per block')
parser.add_argument('--use_cuda', type=bool, default=False, help='offload to gpu')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--mu', type=int, default=128, help='number of epochs')
parser.add_argument('--dataset', type=str, default='ljdataset', help="The dataset to use. Can be 'ljdataset', 'gluon', or 'bach'")
parser.add_argument('--save_path', type=str, default=None, help='path to save trained model')
parser.add_argument('--model_name', type=str, default=None, help='path name of model')
parser.add_argument('--load_path', type=str, default=None, help='path to load saved model')
parser.add_argument('--save_wav', type=str, default=None, help='path to save wav prediction')
parser.add_argument('--test_ratio', type=float, default=0.99, help='ratio of data to use for testing. The rest is used for training.')
parser.add_argument('--gen_len', type=int, default=1, help='number of seconds of data to generate')
parser.add_argument('--stride', type=int, default=500, help='stride to use in bach dataloader.')
parser.add_argument('--save_interval', type=int, default=None, help='Save model every <interval> epochs. Default 0.1 * epochs')


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

def LJ_train(model, optimizer, criterion, evaluate = True):
    model.train()
    print("LJdataset training.")
    epoch_time = timer()
    losses = []
    predictions = None

    '''
    for idx, (x, target) in enumerate(train_loader):
        break
    target = target.view(-1) 
    x, target = x.to(device), target.to(device)
    '''

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
                lss, acc = LJ_evaluate(model, optimizer, criterion)
                print("epoch {} loss {:.3f} acc {:.3f}".format(epoch, lss, acc))

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
    #for idx, (x, target) in enumerate(train_loader):
        #continue
    model.eval()
    x, target = next(iter(train_loader))
    x, target = x.to(device), target.to(device)
    r = model(x).cpu().detach()
    model.train()
    return r

def gluon_train(model, loss_fn, optimizer, mu, seq_size, epochs, batch_size):
    """
    Description : module for running train
    """

    print("Gluon training.")
    loss_save = []
    final_predictions = None
    if (epochs == 0):
        return loss_save, final_predictions
    #fs, data = D.load_wav('parametric-2.wav')
    data = np.load('dataset.npz', mmap_mode='r')
    for key in data.keys():
        print(key)
    data = data['arr_0']
    #data = torch.from_numpy(data).type(torch.LongTensor)
    fs = 16000
    '''
    print(type(data[0]))
    expander = transforms.MuLawExpanding()
    data = expander(data).astype(np.float32)
    to_wav("wav.wav", data, sample_rate=fs)
    quit()
    '''
    print("bach length", len(data))
    g = D.data_generation(data, fs, mu=mu, seq_size=seq_size, dataset = 'gluon')
    best_loss = sys.maxsize
    for epoch in range(epochs):
        loss_tot = 0.0
        predictions = 0
        for i in range(batch_size):
            batch = next(g)
            batch = batch.view(1, -1)
            x = batch[:,:-1]
            x = D.one_hot_encode(x)
            x = torch.tensor(x, dtype=torch.float32)
            x = x.transpose(1, 2)
            x = x.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if i == 0:
                predictions = logits.cpu().detach()
            else:
                predictions = torch.cat((predictions, logits.cpu().detach()), 0)

            sz = logits.shape[0]
            target = batch[0,-sz:]
            loss = loss_fn(logits, target.to(device))
            loss.backward(retain_graph=True)
            loss_tot += loss.item()
            optimizer.step()
        loss_save.append(loss_tot)

        #save the best model
        current_loss = loss_tot
        print('epoch {}, loss {}'.format(epoch, loss_tot))
        if best_loss > current_loss:
            #self.save_model(epoch, current_loss)
            final_predictions = predictions
            best_loss = current_loss
    print("Best loss:", best_loss)
    return loss_save, final_predictions

        
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

    device = torch.device('cpu')
    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device('cuda')
    print("Device:", device)
    model = WaveNet(args.blocks, args.layers_per_block, 24, 256)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.1)
    #optimizer = optim.Rprop(model.parameters()) 
    criterion = nn.CrossEntropyLoss()
    
    if args.load_path:
        state = torch.load(args.load_path)
        model.load_state_dict(state['model'])
        model.to(device)
        optimizer.load_state_dict(state['optimizer'])
    else:
        model.to(device)

    losses = None
    predictions = None 
    mu = 256
    if (args.dataset == 'gluon'):
        ### Gluon train
        seq_size = 20000
        epochs = args.epochs
        batch_size = 64
        losses, predictions = gluon_train(model, criterion, optimizer, mu, seq_size, epochs, batch_size)
    elif args.dataset == 'ljtest':
        D.target_length = 16
        test_data = D.LJDataset(args.data_path, False, args.test_ratio)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=D.collate_fn, num_workers=args.workers)
        lss, acc = LJ_evaluate(model, optimizer, criterion)
        print("loss {:.3f} acc {:.3f}".format(lss, acc))
        quit()
    elif args.dataset == 'bachtest':
        output_length = 16
        test_data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
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
    elif args.dataset == 'bach':
        output_length = 16
        train_data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
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
    model.to(device)
    
    #evaluate
    #avg_loss, accuracy = LJ_evaluate(model, optimizer, criterion)
    #print("Loss per batch {:.3f} accuracy {:.3f}".format(avg_loss, accuracy))

    if args.save_path:
        save_wav = args.save_path + '/' + args.model_name
        D.generation(mu, model, device, filename = save_wav, seconds = args.gen_len, dataset = args.dataset)

        if args.dataset != 'gluon':
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
