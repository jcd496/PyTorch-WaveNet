import os
import torch
import torch.nn as nn
import torch.nn.functional as F
class WaveNet(nn.Module):
    #construct blocks of residual layers. each layer's dilation increases by factor of 2. each layer performs
    #2x1 convolution, separately applies filter and gate, hadamard product results,
    #applies 1x1 convolution, splits output into residual(sum of convolution and input) and split out processing through each layer of each block.
    def __init__(self, num_blocks, residual_layers, hidden_channels, output_channels, global_conditioning, gc_speakers=10):
        super(WaveNet, self).__init__()
        self.num_blocks = num_blocks
        self.residual_layers = residual_layers
        self.global_conditioning = global_conditioning
        
        good_values = [2**layer for layer in range(residual_layers)] 
        self.residual_field = sum(good_values)*num_blocks
        
        self.input_conv=nn.Conv1d(256,hidden_channels,1, bias=False)
        
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
        if global_conditioning:
            gc_filter=[]
            gc_split=[]
        
        for block in range(num_blocks):
            for layer in range(residual_layers):
                dilation = 2**layer
                residual_layers_residual.append(nn.Conv1d(hidden_channels, hidden_channels, 1, bias=False, padding=0, dilation=dilation))
                residual_layers_gated.append(nn.Conv1d(hidden_channels, hidden_channels, 2, bias=False, padding=0, dilation=dilation))
                residual_layers_filter.append(nn.Conv1d(hidden_channels, hidden_channels, 2, bias=False, padding=0, dilation=dilation))
                residual_layers_split.append(nn.Conv1d(hidden_channels, hidden_channels, 1, bias=False, padding=0, dilation=dilation))
                if global_conditioning:
                    gc_filter.append(nn.Linear(gc_speakers, 1, bias=False))
                    gc_split.append(nn.Linear(gc_speakers, 1,  bias=False))
        
        #construct ModuleList of blocks.  describes entire WaveNet
        self.residual = nn.ModuleList(residual_layers_residual)
        self.gated = nn.ModuleList(residual_layers_gated)
        self.filter_ = nn.ModuleList(residual_layers_filter)
        self.split = nn.ModuleList(residual_layers_split)
        
        if global_conditioning:
            self.gc_filter = nn.ModuleList(gc_filter)
            self.gc_split = nn.ModuleList(gc_split)
    
    def forward(self, inputs, conditioning=None):
        
        input_length = inputs.shape[2]
        target_length = input_length - self.residual_field
        #target_length = input_length - self.residual_field - 1 #shift right by single element
        
        residual_out = self.input_conv(inputs)
        final = 0
        if self.global_conditioning:
            condition_vec = conditioning
        
        for block in range(self.num_blocks):
            for layer in range(self.residual_layers):
                dilation = 2**layer
                filter_ = self.filter_[self.residual_layers*block + layer](residual_out)
                gate_ = self.gated[self.residual_layers*block + layer](residual_out)
                
                if self.global_conditioning:
                    filter_ += self.gc_filter[self.residual_layers*block + layer](condition_vec.view(filter_.shape[0],1,-1))
                    gate_ += self.gc_split[self.residual_layers*block + layer](condition_vec.view(gate_.shape[0],1,-1))
                
                filter_ = torch.tanh(filter_)
                gate_ = torch.sigmoid(gate_)
                out = filter_ * gate_
                x = self.residual[self.residual_layers*block + layer](out)
                residual_out = x + residual_out[:,:,-x.shape[2]:]
                
                split_out = self.split[self.residual_layers*block + layer](out)
                #split_out = split_out[:,:,-(target_length+1):-1] #only take elements that have passed through all layers shift right by single element
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

