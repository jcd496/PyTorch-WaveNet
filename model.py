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
    def __init__(self, num_blocks, residual_layers, input_size, hidden_size, output_size, batch_size):
        super(WaveNet, self).__init__()
        self.residual_layers = residual_layers
        self.batch_size = batch_size
        blocks=[]
        self.final_conv1 = nn.Conv1d(in_channels=?,
                                     out_channels=?,
                                     kernel_size=1,
                                     bias=False) #Should bias be true? (vincentherrmann implementation)
        self.final_conv2 = nn.Conv1d(in_channels=?,
                                     out_channesl=?,
                                     kernel_size=1,
                                     bias=False)
        #construct blocks of residual layers
        for block in range(num_blocks):
            #residual_layers_dilated = []
            residual_layers_undilated=[]
            residual_layers_gated=[]
            residual_layers_filter=[]
            #add 2x1 convolution and 1x1 convolution to each residual layer
            for layer in range(residual_layers):
                dilation = 2**layer
                #residual_layers_dilated.append(nn.Conv1d(input_size, hidden_size, 2, padding=0, dilation=dilation))
                #residual_layers_dilated.append(nn.Conv1d(input_size, hidden_size, 2, padding=dilation//2, dilation=dilation))
                residual_layers_undilated.append(nn.Conv1d(hidden_size, output_size, 1, dilation=1))
                residual_layers_gated.append(nn.Conv1d(input_size, hidden_size, 2, padding=0, dilation=dilation))
                residual_layers_filter.append(nn.Conv1d(input_size, hidden_size, 2, padding = 0, dilation=dilation))
            #construct ModuleDict of 2x1 convolution and 1x1 convolution, filter and gate convolutions. describes individual block
            blocks.append(nn.ModuleDict({'undilated':nn.ModuleList(residual_layers_undilated),
                'gated':nn.ModuleList(residual_layers_gated), 'filter':nn.ModuleList(residual_layers_filter)}))
        #construct ModuleList of blocks.  describes entire WaveNet
        self.blocks=nn.ModuleList(blocks)
        self.fc = nn.Linear(15871 - 1023*num_blocks,256)##
        self.softmax = nn.Softmax(dim=1)##
    #helper method for feed forward. describes computation of all residual layers within block    
    @staticmethod
    def residual_layer(inputs, block, residual_layers):
        residual_out = inputs
        for layer in range(residual_layers):
            dilation = 2**layer
            filter_ = block['filter'][layer](residual_out)
            filter_ = torch.tanh(filter_)
            gate_ = block['gated'][layer](residual_out)
            gate_ = torch.sigmoid(gate_)
            x = filter_ * gate_
            x = block['undilated'][layer](x)
            #LEFT PAD ZEROS TO MATCH DIMENSION AFTER 2X1 CONVOLUTION??
            #if(layer==0):
            #    x = F.pad(x, pad=(1,0))
            a = residual_out[:,:,dilation:]
 
            residual_out = x + residual_out[:,:,dilation:]
            split_out = x
        return residual_out, split_out

    def forward(self, inputs):
        residual_out = inputs
        #print(inputs.size())

        final = None
        for i, block in enumerate(self.blocks):
            residual_out, split_out = self.residual_layer(residual_out, self.blocks[i], self.residual_layers)
            #print("residual_out: " + str(residual_out.size()))
            #print("split_out: " + str(split_out.size()))
            if (final == None):
	            final = split_out
            else:
                final = final + split_out


        
        
        fc_out = self.fc(final)##
        fc_out = fc_out.view(-1,256)
        softmax_out = self.softmax(fc_out)
        return softmax_out


#if __name__ == '__main__':
    #just to test
#    model = WaveNet(2, 3, 1, 1, 1)
    #is this the form of input data???
#    inputs = torch.ones([1,1,6])
#    print(model(inputs))
