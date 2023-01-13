import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.special as special
from itertools import combinations
import sys

class MaskedConv3d_h(nn.Conv3d):
    def __init__(self,mask_type, *args, **kwargs):
        super(MaskedConv3d_h, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}  # mask A is for the first convolutional layer only, all deeper layers use mask B
        _, _, kD, kH, kW = self.weight.size()  # get the size of the convolutional filter
        self.register_buffer('mask', self.weight.data.clone())  # initialize mask
        self.mask.fill_(1)  # start building the masks
        #print([kD, kH, kW])
        if mask_type == 'A': # horizontal masking - mask only the central pixel (right-most in a horizontal convolution)
          self.mask[:, :, :, :, -1] = 0  # block right-most pixel in 3D image

    def forward(self,x):
        self.weight.data *= self.mask # apply mask to weight data
        return super(MaskedConv3d_h, self).forward(x)


def gated_activation(input):
    # implement gated activation from Conditional Generation with PixelCNN Encoders
    assert (input.shape[1] % 2) == 0
    a, b = torch.chunk(input, 2, 1) # split input into two equal parts - only works for even number of filters
    a = torch.tanh(a)
    b = torch.sigmoid(b)

    return torch.mul(a,b) # return element-wise (sigmoid-gated) product

class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'gated':
            self.activation = gated_activation
        elif activation_func == 'relu':
            self.activation = F.relu

    def forward(self, input):
        return self.activation(input)


class StackedConvolution(nn.Module):
    def __init__(self, f_in, f_out, kernel_size, padding, dilation, activation, *args, **kwargs):
        super(StackedConvolution, self).__init__(*args, **kwargs)

        self.padding = padding
        self.act_func = activation
        self.pad = dilation * (kernel_size // 2)
        self.d_activation = Activation(self.act_func, f_out)
        self.v_activation = Activation(self.act_func, f_out) # for ReLU, must change number of filters as gated approach halves filters on each application
        self.h_activation = Activation(self.act_func, f_out)
        if activation == 'gated': # filter ratio - need to double a bunch of filters for gated activation
            f_rat = 2
        else:
            f_rat = 1
        self.d_Conv3d = nn.Conv3d(f_in, f_rat * f_out, (kernel_size//2+ 1, kernel_size, kernel_size), 1, padding * (self.pad , self.pad, self.pad),padding_mode='zeros', bias=False)
        self.v_Conv3d = nn.Conv3d(f_in, f_rat * f_out, (1,kernel_size//2 + 1, kernel_size), 1, padding * (0, self.pad, self.pad), bias=False, padding_mode='zeros')
        self.h_Conv3d = nn.Conv3d(f_in, f_rat * f_out, (1,  1, self.pad+1), 1, padding * (0,0, self.pad), bias=False, padding_mode='zeros')
        self.v_to_h_fc = nn.Conv3d(f_rat * f_out, f_rat * f_out, (1, 1, 1), bias=False)
        self.d_to_h_fc = nn.Conv3d(f_rat * f_out, f_rat * f_out, (1, 1, 1), bias=False)
        self.d_to_v_fc = nn.Conv3d(f_rat * f_out, f_rat * f_out, (1, 1, 1), bias=False)


        self.h_to_skip_initial = nn.Conv3d(kernel_size, kernel_size, (1, 1, 1), bias=False)
        #self.h_to_h_initial = nn.Conv3d(kernel_size, kernel_size, (1, 1, 1), bias=False)

        #self.h_Conv2d = nn.Conv2d(f_in, f_rat * f_out, (1, kernel_size // 2 + 1), 1, (0, padding * self.pad), dilation, bias=True, padding_mode='zeros')
        #self.h_to_skip = nn.Conv2d(f_out, f_out, 1, bias=False)
        self.h_to_h = nn.Conv3d(f_out, f_out, (1,1,1), bias=False)

    def forward(self,d_in, v_in, h_in):
        residue = h_in.clone() # residual track

        if self.padding == 0:
            d_in = self.v_Conv3d(d_in) # remove extra padding dstack
            v_in = self.v_Conv3d(v_in) # remove extra padding  vstack
            h_in = self.h_Conv3d(h_in)[:, :, (self.pad):, :-self.pad]  # unpad by 1 on rhs
            residue = residue[:,:,self.pad:,self.pad:-self.pad]
        else:
            d_in= self.d_Conv3d(d_in)[:,:,:-(self.pad),:,:]  #remove extra padding for v to h

            v_in = self.v_Conv3d(v_in)[:, :,:,:-(self.pad),:]  # remove extra padding

            d_to_v= self.d_to_v_fc(d_in)
            d_to_h = self.d_to_h_fc(d_in)  # [:,:,:-1,:] # align v stack to h
            v_to_h = self.v_to_h_fc(v_in)#[:,:,:-1,:] # align v stack to h
            h_in = self.h_Conv3d(h_in)[:, :, :, :,:-self.pad]  # unpad by 1 on rhs

        d_out = self.d_activation(d_in)
        v_out = self.v_activation(torch.add(d_to_v, v_in))
        h_out = (torch.add(d_to_h, h_in))
        h_out = self.h_activation(torch.add(v_to_h, h_out))

        #skip = self.h_to_skip(h_out)
        h_out = self.h_to_h(h_out)
        h_out = torch.add(h_out,residue) # add the residue if the sizes are the same

        return d_out,v_out, h_out#, skip

class GatedPixelCNN(nn.Module):  # Dense or residual, gated, blocked, dilated PixelCNN with batchnorm
    def __init__(self, configs, dataDims):
        super(GatedPixelCNN, self).__init__()

        ### initialize constants
        self.act_func = configs.activation_function
        if self.act_func == 'gated': # filter ratio - need to double a bunch of filters for gated activation
            f_rat = 2
        else:
            f_rat = 1
        kernel_size = configs.conv_size
        initial_convolution_size = configs.conv_size
        self.initial_pad = (initial_convolution_size - 1) // 2
        padding = 1 # DO NOT CHANGE THIS
        channels = dataDims['channels']
        self.layers = configs.conv_layers
        self.filters = configs.conv_filters
        initial_filters = configs.conv_filters


        f_in = (np.ones(configs.conv_layers + 1) * configs.conv_filters).astype(int)
        f_out = (np.ones(configs.conv_layers + 1) * configs.conv_filters).astype(int)
        #Activation
        self.d_init_activation = Activation(self.act_func, initial_filters)
        self.h_init_activation = Activation(self.act_func, initial_filters)
        self.v_init_activation = Activation(self.act_func, initial_filters)
        out_maps = dataDims['classes'] + 1



     #   if configs.do_conditioning:
      #      self.conditioning_fc_1 = nn.Conv2d(dataDims['num conditioning variables'], conditioning_filters, kernel_size=(1, 1))
      #  self.conditioning_fc_2 = nn.Conv2d(conditioning_filters, initial_filters, kernel_size=(1, 1))

        # initial layer
        self.d_initial_convolution = nn.Conv3d(channels, f_rat * initial_filters,(self.initial_pad + 1, initial_convolution_size, initial_convolution_size), 1, padding * (self.initial_pad+1, self.initial_pad , self.initial_pad),
                                               dilation=1, padding_mode='zeros', bias=False)

        self.v_initial_convolution = nn.Conv3d(channels, f_rat * initial_filters, (1,self.initial_pad + 1, initial_convolution_size), 1, padding * (0,self.initial_pad + 1, self.initial_pad),dilation=1, padding_mode='zeros', bias=False)

        self.h_initial_convolution = MaskedConv3d_h('A',  channels, f_rat * initial_filters,(1,1, self.initial_pad + 1), 1, padding * (0,0, self.initial_pad),dilation=1,
                                                    padding_mode='zeros', bias=False)

        self.v_to_h_initial = nn.Conv3d(f_rat * initial_filters, f_rat * initial_filters, (1, 1, 1), bias=False)
        self.d_to_h_initial = nn.Conv3d(f_rat * initial_filters, f_rat * initial_filters, (1, 1, 1), bias=False)
        self.d_to_v_initial = nn.Conv3d(f_rat * initial_filters, f_rat * initial_filters, (1, 1, 1), bias=False)

        self.h_to_skip_initial = nn.Conv3d(initial_filters, initial_filters, (1,1,1), bias=False)
        self.h_to_h_initial = nn.Conv3d(initial_filters, initial_filters, (1,1,1), bias=False)

        # stack hidden layers in blocks
        self.conv_layer = [StackedConvolution(f_in[i], f_out[i], kernel_size, padding, 1, self.act_func) for i in range(configs.conv_layers)] # stacked convolution (no blind spot)
        self.conv_layer = nn.ModuleList(self.conv_layer)

        #output layers
        fc_filters = configs.conv_filters
        self.fc_activation = Activation('relu', fc_filters)# // f_rat)
        self.fc1 = nn.Conv3d(f_out[-1], fc_filters, (1, 1,1), bias=True)  # add skip connections
        self.fc2 = nn.Conv3d(fc_filters, out_maps * channels, (1,1,1), bias=False) # gated activation cuts filters by 2
        self.fc_norm = nn.BatchNorm3d(fc_filters)
        self.fc_dropout = nn.Dropout(configs.fc_dropout_probability)

    def forward(self, input):
        # clean input

        # initial convolution
        d_data= self.d_initial_convolution(input)[:, :, :-(self.initial_pad + 2), :,:]  # remove extra
        #print([d_data.shape,'d shape conv'])
        d_to_v_data = self.d_to_v_initial(d_data)  # [:,:,:-1,:] # align with v-stack

        v_data = self.v_initial_convolution(input)[:, :, :, :-(self.initial_pad + 2), :]  # remove extra
        v_to_h_data = self.v_to_h_initial(v_data)#[:,:,:-1,:] # align with h-stack
        d_to_h_data = self.d_to_h_initial(d_data)  # [:,:,:-1,:] # align with h-stack

        h_data = self.h_initial_convolution(input)[:,:,:,:,:-self.initial_pad] # unpad rhs of image
        d_data = self.d_init_activation(d_data)
        v_data = self.v_init_activation(d_to_v_data + v_data)
        h_data = self.h_init_activation(v_to_h_data+d_to_h_data + h_data)



        h_data = self.h_to_h_initial(h_data)

        # hidden layers
        for i in range(self.layers):
            d_data, v_data, h_data = self.conv_layer[i](d_data,v_data, h_data) # stacked convolutions fix blind spot

        # output convolutions
        x = self.fc1(h_data)
        x = self.fc_norm(x)
        x = self.fc_activation(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)

        return x

