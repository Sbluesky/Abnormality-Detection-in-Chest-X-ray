import os
import numpy as np
from typing import Union
import uuid

from tensorflow.keras import Model, Sequential
from tensorflow.keras import activations
from tensorflow.keras.layers import (Input, Add, Conv2D, GlobalAveragePooling2D, 
                                     GlobalMaxPooling2D,MaxPooling2D, Dropout,
                                     UpSampling2D, Concatenate, Conv2DTranspose,
                                     BatchNormalization, Dense, Activation, MaxPool2D, AveragePooling2D)
                                     

def genID():
    return str(uuid.uuid4())[-3:]
    
class UshouldRestNet(Model):
    def __init__(self,
                tensor_shape: tuple,
                num_class: int, 
                model_config: dict):
        
        super(UshouldRestNet, self).__init__()
        
        #General Model config
        self.tensor_shape = tensor_shape
        self.num_class = num_class
        
        #Encoder config
        self.num_conv_blocks  = model_config['num_conv_blocks']
        self.conv_kernel_size = model_config['conv_kernel_size']
        self.num_conv_kernel  = model_config['num_conv_kernel']
        self.conv_strides     = model_config['conv_strides']
        self.mode             = model_config['mode']
        self.pooling          = model_config['pooling']
        self.dropout_rate     = model_config['dropout_rate']
        
        #Decoder config
        self.encoder_chains   = []
        self.upsampling_method= model_config['upsampling_method']
        
    def _conv_batchnorm_dropout_(
                    self,
                    _input_, 
                    filters: int,
                    kernel_size: Union[int, tuple] = (3,3),
                    strides : Union[tuple, int] = (1,1),
                    padding: str = 'same',
                    kernel_initializer='he_normal'
                          ):
        output = Conv2D(filters = filters, 
                        kernel_size = kernel_size, 
                        strides = strides,padding = padding, 
                        kernel_initializer=kernel_initializer, 
                        name = f'ksize{kernel_size[0]}_stride{strides[0]}_{genID()}')(_input_)
        output = BatchNormalization()(output)
        output = Dropout(self.dropout_rate)(output)
        return output

        
    def _small_conv_block_(self, 
                    _input_, 
                    filters: int,
                    kernel_size: Union[int, tuple] = (3,3),
                    strides : Union[tuple, int] = (1,1),
                    padding: str = 'same',
                    position: str = 'non_first_block'
                          ):
        '''
        Defining a small 2_consecutive_conv blocks
        Params:
        :filters: number of filters in current blocks
        :kernel_size:    size of the kernel in conv_layer
        :stride:         stride of conv_kernel
        :padding:        default padding for conv_layer
        :position:       position of these 2 conv_blocks with respect to the Net
        Return:
        Output of the 2 conv_blocks
        '''
        #Depend on the position
        if(position == 'very_first_block'):
            #Very first block of all Resnet
            _input_ = self._conv_batchnorm_dropout_(_input_,filters = 64, kernel_size = (7,7), strides = (2,2), padding = 'same')
            _input_ = Activation('relu')(_input_)
            _input_ = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(_input_)

            #Just for the second conv_block of every Resnet, strides = (1,1)            
            stride_first_layer = strides
            
        elif(position == 'first_block'):
            #For every first conv_block of larger conv_chain
            stride_first_layer = (2,2)
            
        elif(position == 'non_first_block'):
            #Else
            stride_first_layer = strides      
        
        input_shape = _input_.shape
        #A module of 2 conv_blocks
        if(self.mode == '2in1'):
            output = self._conv_batchnorm_dropout_(_input_,filters = filters, strides = stride_first_layer)
            output = Activation('relu')(output)
            output_shape = output.shape
            output = self._conv_batchnorm_dropout_(output, filters = filters, strides = strides)
            
            
        elif(self.mode == '3in1'):
            pass
            return

        #Check to see what kind of skip connection is used
        if(input_shape[1] == output_shape[1] and input_shape[-1] == output_shape[-1]):
        #if(input_shape == output_shape): Don't know why this doesn't work
            shortcut = _input_
        else:
            shortcut = Conv2D(filters, kernel_size = (1, 1), strides=(2,2),padding = 'same',
                             kernel_initializer='he_normal')(_input_)
            shortcut = BatchNormalization()(shortcut)
            shortcut = Dropout(self.dropout_rate)(shortcut)
        
        output = Add()([output, shortcut]) #Shortcut is merged
        output = Activation('relu')(output)
        
        return output
           
        
    def _large_conv_block_(self, 
                    _input_, 
                    filters: int,
                    num_conv_blocks: int,
                    kernel_size: Union[int, tuple] = (3,3),
                    strides : Union[tuple, int] = (1,1),
                    padding: str = 'same',
                    position: str = 'non_first_block'
                          ):
        '''
        Defining a large chain of consecutive conv blocks with the same number of filters
        Params:
        :filters:        number of filters in current blocks
        :num_conv_blocks:number of conv_blocks in 1 chain
        :kernel_size:    size of the kernel in conv_layer
        :stride:         stride of conv_kernel
        :padding:        default padding for conv_layer
        :position:       position of these 2 conv_blocks with respect to the Net
        Return:
        Output of the chain of conv_blocks
        '''
#         if(num_conv_blocks % 2 == 1):
#             raise ValueError('num_conv_blocks has to have even number of blocks')
            
        #If the position of the chain is at the beginning, strides are different
        if(position == 'very_first_chain'):
            output = self._small_conv_block_(_input_, filters = filters, position = 'very_first_block')
        else:    
            output = self._small_conv_block_(_input_, filters = filters, position = 'first_block')
        
        #For the rest of the chain, strides are all (1,1)
        for index in range(num_conv_blocks - 1):
            output = self._small_conv_block_(output, filters = filters, position = 'non_first_block')
        
        return output
    
    #Decoding
    def UpSampling(self, _input_, kernel = (2,2)):
        return UpSampling2D(size = kernel)(_input_)
    
    def InvTranspose(self, _input_, filters):
        return Conv2DTranspose(filters = filters, kernel_size = (3,3), padding = 'same', strides = (2,2))(_input_)
        
    def _decode_block_(self,
                     _input_,
                     filters):
        if(self.upsampling_method == 'UpSampling'):
            output = self.UpSampling(_input_)
            output = Conv2D(filters, kernel_size = (2,2), strides = (1,1), padding = 'same')(output)
            
        elif(self.upsampling_method == 'TransposeConv'):
            output = self.InvTranspose(_input_, filters)
        
        else:
            raise ValueError('Invalid error for method of upsampling')
            
#         output = self._conv_batchnorm_dropout_(_input_,filters = filters, strides = stride_first_layer)
        return output
        
    def call(self, _input_):
        #Going down
        output = self._large_conv_block_(_input_, 
                                               filters = self.num_conv_kernel[0],
                                               position = 'very_first_chain',
                                               num_conv_blocks = self.num_conv_blocks[0])
        print(output.shape)
        self.encoder_chains.append(output)
        
        for index in range(1, len(self.num_conv_kernel)):
            output = self._large_conv_block_(output, 
                                               filters = self.num_conv_kernel[index],
                                               num_conv_blocks = self.num_conv_blocks[index])
#             print(output.shape)
            self.encoder_chains.append(output)
        
        #Middle
        output = self._conv_batchnorm_dropout_(output,filters = self.num_conv_kernel[-1]*2, strides = (2,2), padding = 'same')
        output = Activation('relu')(output)
        
        #Going up
        
        for index, encoder_output in enumerate(self.encoder_chains[::-1]):
#             print('be4',output.shape)
#             print(self.num_conv_kernel[::-1][index])
            output = self._decode_block_(output, filters = self.num_conv_kernel[::-1][index])
#             print('after decode',output.shape)
            output = Concatenate(axis = -1)([output, encoder_output])
#             print('after concat',output.shape)
        
            output = self._conv_batchnorm_dropout_(output,filters = self.num_conv_kernel[::-1][index])
            output = self._conv_batchnorm_dropout_(output,filters = self.num_conv_kernel[::-1][index])
        
        prediction = Conv2D(name="PredictionMask",
                             filters=self.num_class, kernel_size=(1, 1),
                             activation="sigmoid")(output)
        return prediction
    
    def create(self):
        _input_ = Input(shape=self.tensor_shape)
        output = self.call(_input_)
        return Model(inputs=_input_, outputs=output)

_34 = {
    'name'             : 'Resnet34',
    'num_conv_blocks'  : [3,4,6,3],
    'conv_kernel_size' : 3,
    'num_conv_kernel'  : [64, 128, 256, 512],
    'conv_strides'     : 1,
    'mode'             : '2in1',
    
    'dropout_rate'     : 0.0,
    'pooling'          : 'max',
    'upsampling_method': 'UpSampling'
}

_18 = {
    'name'             : 'Resnet18',
    'num_conv_blocks'  : [2,2,2,2],
    'conv_kernel_size' : 3,
    'num_conv_kernel'  : [64, 128, 256, 512],
    'conv_strides'     : 1,
    'mode'             : '2in1',
    
    'dropout_rate'     : 0.0,
    'pooling'          : 'max',
    'upsampling_method': 'UpSampling'
}

#Calling Resnet18_Unet
model = UshouldRestNet(tensor_shape = (512,512,3), num_class = 12, model_config = _18).create()
model.summary()

#Calling Resnet34_Unet
model = UshouldRestNet(tensor_shape = (512,512,3), num_class = 12, model_config = _34).create()
model.summary()