import os
import torch
import torch.nn as nn
import torch.nn.functional as F
class WaveNet(nn.Module):
    #construct blocks of residual layers. each layer's dilation increases by factor of 2. each layer performs
    #2x1 convolution, separately applies filter and gate, hadamard product results,
    #applies 1x1 convolution, splits output into residual(sum of convolution and input) and split out processing through each layer of each block.
    #blocks are ModuleList of ModuleDict's, each ModuleDict contains two ModuleList, dilated convolution layers and undilated convolution layer.
    #this ensures modules are registered and visable by all methods.
    def __init__(self, num_blocks, residual_layers, hidden_channels, output_channels, batch_size):
        super(WaveNet, self).__init__()
        self.residual_layers = residual_layers
        self.batch_size = batch_size
        good_values = [2**layer for layer in range(residual_layers)]
        self.good_values = sum(good_values)*num_blocks
        blocks=[]
        self.input_conv=nn.Conv1d(256,hidden_channels,1)
        self.final_conv1 = nn.Conv1d(in_channels=hidden_channels,
                                     out_channels=512,
                                     kernel_size=1,
                                     bias=False) #Should bias be true? (vincentherrmann implementation)
        self.final_conv2 = nn.Conv1d(in_channels=512,
                                     out_channels=output_channels,
                                     kernel_size=1,
                                     bias=False)
        #construct blocks of residual layers
        for block in range(num_blocks):
            residual_layers_residual=[]
            residual_layers_gated=[]
            residual_layers_filter=[]
            residual_layers_split=[]
            #add 2x1 convolution and 1x1 convolution to each residual layer
            for layer in range(residual_layers):
                dilation = 2**layer
                residual_layers_residual.append(nn.Conv1d(hidden_channels, hidden_channels, 1, dilation=dilation))
                residual_layers_gated.append(nn.Conv1d(hidden_channels, hidden_channels, 2, padding=0, dilation=dilation))
                residual_layers_filter.append(nn.Conv1d(hidden_channels, hidden_channels, 2, padding = 0, dilation=dilation))
                residual_layers_split.append(nn.Conv1d(hidden_channels, 512, 1, padding=0, dilation=dilation))
            #construct ModuleDict of 2x1 convolution and 1x1 convolution, filter and gate convolutions. describes individual block
            blocks.append(nn.ModuleDict({'residual':nn.ModuleList(residual_layers_residual),
                                        'gated':nn.ModuleList(residual_layers_gated),
                                        'filter':nn.ModuleList(residual_layers_filter), 
                                        'split': nn.ModuleList(residual_layers_split)}))

        #construct ModuleList of blocks.  describes entire WaveNet
        self.blocks=nn.ModuleList(blocks)
        self.softmax = nn.Softmax(dim=1)##
    #helper method for feed forward. describes computation of all residual layers within block    
    
    def residual_layer(self, inputs, block):
        residual_out = inputs
        for layer in range(self.residual_layers):
            dilation = 2**layer
            filter_ = block['filter'][layer](residual_out)
            filter_ = torch.tanh(filter_)
            gate_ = block['gated'][layer](residual_out)
            gate_ = torch.sigmoid(gate_)
            out = filter_ * gate_
            x = block['residual'][layer](out)
            residual_out = x + residual_out[:,:,dilation:]
            split_out = block['split'][layer](out)
            split_out = x[:,:,self.good_values:] #only take elements that have passed through all layers
        return residual_out, split_out

    def forward(self, inputs):
        residual_out = self.input_conv(inputs)
        final = None
        for i, block in enumerate(self.blocks):
            residual_out, split_out = self.residual_layer(residual_out, self.blocks[i])
            if (final == None):
	            final = split_out
            else:
                final = final + split_out
        x = torch.relu(final)
        x = self.final_conv1(x)
        x = torch.relu(x)
        x = self.final_conv2(x)
        #x = torch.transpose(x, 2, 1)
        x = torch.sum(x, dim=2)/x.shape[2]
        x = self.softmax(x)
        return x


