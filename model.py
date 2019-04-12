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
    def __init__(self, num_blocks, residual_layers, input_size, hidden_size, output_size):
        super(WaveNet, self).__init__()
        self.residual_layers = residual_layers
        blocks=[]
        #construct blocks of residual layers
        for block in range(num_blocks):
            residual_layers_dilated = []
            residual_layers_undilated=[]
            #add 2x1 convolution and 1x1 convolution to each residual layer
            for layer in range(residual_layers):
                dilation = 2**layer
                residual_layers_dilated.append(nn.Conv1d(input_size, hidden_size, 2, padding=dilation//2, dilation=dilation))
                residual_layers_undilated.append(nn.Conv1d(hidden_size, output_size, 1, dilation=1))
            #construct ModuleDict of 2x1 convolution and 1x1 convolution. describes individual block
            blocks.append(nn.ModuleDict({'dilated':nn.ModuleList(residual_layers_dilated), 'undilated':nn.ModuleList(residual_layers_undilated)}))
        #construct ModuleList of blocks.  describes entire WaveNet
        self.blocks=nn.ModuleList(blocks)
        self.softmax = nn.Softmax()
    #helper method for feed forward. describes computation of all residual layers within block    
    @staticmethod
    def residual_layer(inputs, block, residual_layers):
        residual_out = inputs
        for layer in range(residual_layers):
            x = block['dilated'][layer](residual_out)
            filter_ = torch.tanh(x)
            gate_ = torch.sigmoid(x)
            x = filter_ * gate_
            x = block['undilated'][layer](x)
            #LEFT PAD ZEROS TO MATCH DIMENSION AFTER 2X1 CONVOLUTION??
            if(layer==0):
                x = F.pad(x, pad=(1,0))
            residual_out = x + residual_out
            split_out = x
        return residual_out, split_out

    def forward(self, inputs):
        residual_out = inputs
        for i, block in enumerate(self.blocks):
            residual_out, split_out = self.residual_layer(residual_out, self.blocks[i], self.residual_layers)
        softmax_out = self.softmax(residual_out)
        return softmax_out


#if __name__ == '__main__':
    #just to test
#    model = WaveNet(2, 3, 1, 1, 1)
    #is this the form of input data???
#    inputs = torch.ones([1,1,6])
#    print(model(inputs))
