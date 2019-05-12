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

        residual_channels = 32
        dilation_channels = 32
        skip_channels = 1024
        end_channels = 512
        self.target_length = None

        good_values = [2**layer for layer in range(residual_layers)] 
        self.residual_field = sum(good_values)*num_blocks
        
        self.input_conv=nn.Conv1d(256,residual_channels,1, bias=False)
        self.final_conv1 = nn.Conv1d(in_channels=skip_channels,
                                     out_channels=end_channels,
                                     kernel_size=1,
                                     bias=True) #Should bias be true? (vincentherrmann implementation)
        self.final_conv2 = nn.Conv1d(in_channels=end_channels,
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

                residual_layers_gated.append(nn.Conv1d(
                                                in_channels=residual_channels, 
                                                out_channels=dilation_channels, 
                                                kernel_size=2, bias=False, padding=dilation, dilation=dilation))
                residual_layers_filter.append(nn.Conv1d(
                                                in_channels=residual_channels, 
                                                out_channels=dilation_channels, 
                                                kernel_size=2, bias=False, padding=dilation, dilation=dilation))

                residual_layers_residual.append(nn.Conv1d(
                                                in_channels=dilation_channels, 
                                                out_channels=residual_channels, 
                                                kernel_size=1, bias=False, padding=0))
                residual_layers_split.append(nn.Conv1d(
                                                in_channels=dilation_channels, 
                                                out_channels=skip_channels, 
                                                kernel_size=1, bias=False, padding=0))

        #construct ModuleList of blocks.  describes entire WaveNet
        self.gated = nn.ModuleList(residual_layers_gated)
        self.filter_ = nn.ModuleList(residual_layers_filter)
        self.residual = nn.ModuleList(residual_layers_residual)
        self.split = nn.ModuleList(residual_layers_split)
        
        self.softmax = nn.LogSoftmax(dim=1)##
    
    def forward(self, inputs):
        input_length = inputs.shape[2]
        target_length = input_length - self.residual_field
        self.target_length = target_length
        residual_out = self.input_conv(inputs)
        final = 0
        for block in range(self.num_blocks):
            for layer in range(self.residual_layers):
                dilation = 2**layer
                filter_ = self.filter_[self.residual_layers*block + layer](residual_out)
                filter_ = filter_[:,:,:-dilation]
                filter_ = torch.tanh(filter_)
                gate_ = self.gated[self.residual_layers*block + layer](residual_out)
                gate_ = gate_[:,:,:-dilation]
                gate_ = torch.sigmoid(gate_)
                out = filter_ * gate_
                x = self.residual[self.residual_layers*block + layer](out)
                residual_out = x + residual_out[:,:,-x.shape[2]:]
                split_out = self.split[self.residual_layers*block + layer](out)
                split_out = split_out[:,:,-target_length:] #only take elements that have passed through all layers
                final = final + split_out

        x = torch.relu(final)
        x = self.final_conv1(x)
        x = torch.relu(x)
        x = self.final_conv2(x)

        [n, c, l] = x.size()
        x = x.transpose(1, 2).contiguous()
        x = x.view(n * l, c)
        return x

