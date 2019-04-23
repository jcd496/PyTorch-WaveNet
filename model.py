import os
import torch
import torch.nn as nn
import torch.nn.functional as F
class WaveNet(nn.Module):
    #construct blocks of residual layers. each layer's dilation increases by factor of 2. each layer performs
    #2x1 convolution, separately applies filter and gate, hadamard product results,
    #applies 1x1 convolution, splits output into residual(sum of convolution and input) and split out processing through each layer of each block.
    def __init__(self, num_blocks, residual_layers, hidden_channels, output_channels):
        super(WaveNet, self).__init__()
        self.num_blocks = num_blocks
        self.residual_layers = residual_layers

        good_values = [2**layer for layer in range(residual_layers)]
        self.good_values = sum(good_values)*num_blocks
        self.receptive_field = self.good_values
        self.skip_start = self.good_values - 1
        
        self.input_conv=nn.Conv1d(256,hidden_channels,1)
        self.final_conv1 = nn.Conv1d(in_channels=hidden_channels,
                                     out_channels=512,
                                     kernel_size=1,
                                     bias=True) #Should bias be true? (vincentherrmann implementation)
        self.final_conv2 = nn.Conv1d(in_channels=512,
                                     out_channels=output_channels,
                                     kernel_size=1,
                                     bias=True)

        #construct blocks of residual layers

        residual_layers_residual=[]
        residual_layers_gated=[]
        residual_layers_filter=[]
        residual_layers_split=[]
        for block in range(num_blocks):
            #add 2x1 convolution and 1x1 convolution to each residual layer
            for layer in range(residual_layers):
                dilation = 2**layer
                residual_layers_residual.append(nn.Conv1d(hidden_channels, hidden_channels, 1, padding=0, dilation=dilation))
                residual_layers_gated.append(nn.Conv1d(hidden_channels, hidden_channels, 2, padding=0, dilation=dilation))
                residual_layers_filter.append(nn.Conv1d(hidden_channels, hidden_channels, 2, padding = 0, dilation=dilation))
                residual_layers_split.append(nn.Conv1d(hidden_channels, hidden_channels, 1, padding=0, dilation=dilation))
            #construct ModuleDict of 2x1 convolution and 1x1 convolution, filter and gate convolutions. describes individual block

        #construct ModuleList of blocks.  describes entire WaveNet
        self.residual = nn.ModuleList(residual_layers_residual)
        self.gated = nn.ModuleList(residual_layers_gated)
        self.filter = nn.ModuleList(residual_layers_filter)
        self.split = nn.ModuleList(residual_layers_split)
        
        self.softmax = nn.Softmax(dim=1)##
    #helper method for feed forward. describes computation of all residual layers within block    
    
    def forward(self, inputs):
        input_length = inputs.shape[2]
        target_length = input_length - self.good_values
        residual_out = self.input_conv(inputs)
        final = 0
        final_set = False
        for block in range(self.num_blocks):
            for layer in range(self.residual_layers):
                dilation = 2**layer
                filter_ = self.filter[self.residual_layers*block + layer](residual_out)
                filter_ = torch.tanh(filter_)
                gate_ = self.gated[self.residual_layers*block + layer](residual_out)
                gate_ = torch.sigmoid(gate_)
                out = filter_ * gate_
                x = self.residual[self.residual_layers*block + layer](out)
                residual_out = x + residual_out[:,:,dilation:]
                split_out = self.split[self.residual_layers*block + layer](out)
                split_out = x[:,:,-target_length:] #only take elements that have passed through all layers
                if final_set:
                    final = final + split_out
                else:
                    final_set = True
                    final = split_out

        x = torch.relu(final)
        x = self.final_conv1(x)
        x = torch.relu(x)
        x = self.final_conv2(x)

        [n, c, l] = x.size()
        x = x.transpose(1, 2).contiguous()
        x = x.view(n * l, c)
        #print(x.shape)
        #x = torch.transpose(x, 2, 1)
        #x = torch.sum(x, dim=2)/x.shape[2]
        #print(x.shape)
        #print(x)
        #x = self.softmax(x)
        #print(x.shape)
        #print(x)
        #§§quit()
        #x = x[:,:,-1023:]
        #print("Output dimensions:", x.shape)
        return x

