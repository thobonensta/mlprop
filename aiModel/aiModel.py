#----------------------------------------------------------------------------------------------------------------------#
#                              FILE CONTAINING THE AI ARCHITECTURE TO COMPUTES 1D PL                                   #
#                  The intenting goal is to generalize this method to 2D so as to create a deep-SSW                    #
#                           The chosen architecture is a U-NET with 4 layers                                           #
#----------------------------------------------------------------------------------------------------------------------#
# Import (the AI model is build with torch)
import numpy as np
import torch
import torch.nn as nn

#----------------------------------------------------------------------------------------------------------------------#
# Convolution block
#----------------------------------------------------------------------------------------------------------------------#
class conv_block(nn.Module):
    '''  Class that defines the convolution block used in the U-Net architecture
    The basic convolution block consists of on conv1D, one batch norm and one ReLU activation
    This latter is performed twice
    Instances : in_c, out_c, dilation, kernel_size (number of parameters to optimize, size of the conv window), padding
    Object : one convolution block
    '''

    def __init__(self, in_c, out_c,kernelSize=3,dilation=1):
        ''' Initialization of the object conv_block'''
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=kernelSize,dilation=dilation, stride=1,padding="same")
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=kernelSize,dilation=dilation, stride=1,padding="same")
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        ''' Method that computes one forward pass through the conv block'''
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

#----------------------------------------------------------------------------------------------------------------------#
# Encoder part : the objective is to search for the features at different levels
#----------------------------------------------------------------------------------------------------------------------#
class encoder_block(nn.Module):
    ''' Class that creates the encoder layers '''
    def __init__(self, in_c, out_c, kernelSize=3,dilation=1):
        super().__init__()
        self.conv = conv_block(in_c, out_c, kernelSize,dilation) # convolution
        self.pool = nn.MaxPool1d(2) # max pooling
    def forward(self, inputs):
        x = self.conv(inputs) # used for the direct forward
        p = self.pool(x) # used to go into the following layer
        return x, p

#----------------------------------------------------------------------------------------------------------------------#
# Decoder part : from these features we can compute a path loss
#----------------------------------------------------------------------------------------------------------------------#
class decoder_block(nn.Module):
    ''' Class that creates the decoder layers '''
    def __init__(self, in_c, out_c,kernelSize=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=kernelSize, stride=2, padding=0) # Upsampling via transpose convolution
        self.conv = conv_block(out_c+out_c, out_c) # convolution layer
    def forward(self, inputs, skip):
        x = self.up(inputs) # Go up
        skip = skip[:,:,:x.shape[2]]
        x = torch.cat([x, skip], axis=1) # Concatenate with the direct forward convolve
        x = self.conv(x) # Convolve
        return x

#----------------------------------------------------------------------------------------------------------------------#
# U-Net architecture : composed of the encoder and decoder
#----------------------------------------------------------------------------------------------------------------------#
class build_unet(nn.Module):
    ''' class that creates the U-Net using the encoder and decoder block
    diffI allows to increased the dilation throughout the layers to add
    more informations to the lower stages increasing the accuracy
    '''
    def __init__(self,kernelSizeConv=3,kernelSizeUp=2,dilation=1,diffI=False,p=0):
        super().__init__()
        self.dilation = dilation
        self.do = nn.Dropout(p=p)

        if diffI: # The idea is to add more spatial features when decreasing the size
            """ Encoder """
            self.e1 = encoder_block(1, 64,kernelSizeConv,self.dilation)
            self.e1.requires_grad_(True)
            self.e2 = encoder_block(64, 128,kernelSizeConv,self.dilation)
            self.e2.requires_grad_(True)
            self.e3 = encoder_block(128, 256,kernelSizeConv,self.dilation*2)
            self.e3.requires_grad_(True)
            self.e4 = encoder_block(256, 512,kernelSizeConv,self.dilation*2)
            self.e4.requires_grad_(True)
            """ Bottleneck """
            self.b = conv_block(512, 1024, kernelSizeConv,self.dilation*2)
            self.b.requires_grad_(True)
        else:
            """ Encoder """
            self.e1 = encoder_block(1, 64,kernelSizeConv,self.dilation)
            self.e2 = encoder_block(64, 128,kernelSizeConv,self.dilation)
            self.e3 = encoder_block(128, 256,kernelSizeConv,self.dilation)
            self.e4 = encoder_block(256, 512,kernelSizeConv,self.dilation)
            """ Bottleneck """
            self.b = conv_block(512, 1024,kernelSizeConv, self.dilation)
        """ Decoder """
        self.d1 = decoder_block(1024, 512,kernelSizeUp)
        self.d1.requires_grad_(True)
        self.d2 = decoder_block(512, 256,kernelSizeUp)
        self.d2.requires_grad_(True)
        self.d3 = decoder_block(256, 128,kernelSizeUp)
        self.d3.requires_grad_(True)
        self.d4 = decoder_block(128, 64,kernelSizeUp)
        self.d4.requires_grad_(True)
        """ Output """
        self.outputs = nn.Conv1d(64, 1, kernel_size=1, padding=0)
        self.outputs.requires_grad_(True)
        # The bias is set at first manually to the mean of the field to accelerate the training
        self.outputs.bias.data = torch.from_numpy(-25.9*np.ones(1))
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        p3 = self.do(p3)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        p4 = self.do(p4)
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d1 = self.do(d1)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)

        return outputs
