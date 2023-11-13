import os
import configparser as cp

import torch
import torch.nn as nn
import torch.nn.functional as F

from weight_rebuild.non_ideal.SAF import SAF_mask_gen
from weight_rebuild.non_ideal.rand_noise import rand_noise_gen
from weight_rebuild.weight_mapping.weight_reshape import cxb_count_2d
from weight_rebuild.weight_mapping.weight_reshape import weight_to_rram
from weight_rebuild.weight_mapping.weight_reshape import rram_to_weight

configs = cp.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config.ini')
configs.read(config_path)

data_type = getattr(torch,configs.get('DEFAULT', 'precision')) 

class cxb_linear(nn.Linear):

    def __init__(
        self, 
        in_features,
        out_features,
        weight,
        bias,
        device=torch.device('cuda:0'),
        cxb_shape=(64,64),
        min_g=3e-6,
        max_g=3e-3,
        wire_g=3e-4,
        SA0_ratio=0.1,
        SA1_ratio=0.1,
        en_nonidealities=[0,0,0,0,0,0]
        ):
        super(cxb_linear, self).__init__(in_features,out_features,bias)

        self.device=device
        self.cxb_shape=cxb_shape
        self.min_g=min_g
        self.max_g=max_g
        self.wire_g=wire_g
        self.en_nonidealities=en_nonidealities
        self.weight.data=weight.to(self.device)
        
        rram_weight=weight_to_rram(self.weight,self.cxb_shape).to(self.device)

        self.pos_min_g=torch.amin(torch.relu(rram_weight),dim=(2,3),keepdim=True)
        self.pos_max_g=torch.amax(torch.relu(rram_weight),dim=(2,3),keepdim=True)
        self.neg_min_g=torch.amin(torch.relu(-rram_weight),dim=(2,3),keepdim=True)
        self.neg_max_g=torch.amax(torch.relu(-rram_weight),dim=(2,3),keepdim=True)

        self.SA0_ratio=SA0_ratio
        self.SA1_ratio=SA1_ratio
        
        weight_mapped_shape,self.pad_shape=cxb_count_2d(self.weight.shape,self.cxb_shape)
        self.weight_mapped_shape=(weight_mapped_shape[0],weight_mapped_shape[2],weight_mapped_shape[1],weight_mapped_shape[3])

        self.pos_SA0_mask,self.pos_SA1_mask,self.neg_SA0_mask,self.neg_SA1_mask=SAF_mask_gen(
            self.weight.shape,
            self.weight_mapped_shape,
            self.SA0_ratio,
            self.SA1_ratio,
            self.device)
        
    def forward(self,input):

        pos_conv_weight=torch.relu(self.weight).to(self.device)
        neg_conv_weight=torch.relu(-self.weight).to(self.device)

        if self.en_nonidealities[0]:

            pos_conv_weight = \
                pos_conv_weight*((torch.ones(self.weight.shape,device=self.device)-(self.pos_SA0_mask+self.pos_SA1_mask)).detach())\
                +rram_to_weight(
                    self.pos_max_g*torch.ones(self.weight_mapped_shape,device=self.device),
                    self.weight.shape,self.cxb_shape)*self.pos_SA0_mask.detach()\
                +rram_to_weight(
                    self.pos_min_g*torch.ones(self.weight_mapped_shape,device=self.device),
                    self.weight.shape,self.cxb_shape)*self.pos_SA1_mask.detach()
            neg_conv_weight = \
                neg_conv_weight*((torch.ones(self.weight.shape,device=self.device)-(self.neg_SA0_mask+self.neg_SA1_mask)).detach())\
                +rram_to_weight(
                    self.neg_max_g*torch.ones(self.weight_mapped_shape,device=self.device),
                    self.weight.shape,self.cxb_shape)*self.neg_SA0_mask.detach()\
                +rram_to_weight(
                    self.neg_min_g*torch.ones(self.weight_mapped_shape,device=self.device),
                    self.weight.shape,self.cxb_shape)*self.neg_SA1_mask.detach()
        
        rand_noise_sum = sum(self.en_nonidealities[1:])
        if (rand_noise_sum > 0):
            pos_conv_weight = pos_conv_weight + \
                rand_noise_gen(weight=pos_conv_weight,
                               min_w=self.pos_min_g,
                               max_w=self.pos_max_g,
                               en_thermal_noise=self.en_nonidealities[1],
                               en_shot_noise=self.en_nonidealities[2],
                               en_rtn = self.en_nonidealities[3],
                               en_prog_error=self.en_nonidealities[4],
                               device=self.device).detach()
            neg_conv_weight = neg_conv_weight + \
                rand_noise_gen(weight=neg_conv_weight,
                               min_w=self.neg_min_g,
                               max_w=self.neg_max_g,
                               en_thermal_noise=self.en_nonidealities[1],
                               en_shot_noise=self.en_nonidealities[2],
                               en_rtn = self.en_nonidealities[3],
                               en_prog_error=self.en_nonidealities[4],
                               device=self.device).detach()

        
        pos_conv_weight = pos_conv_weight.type(data_type)
        neg_conv_weight = neg_conv_weight.type(data_type)  
        
        pos_result=F.linear(input,pos_conv_weight,self.bias)
        neg_result=F.linear(input=input,weight=neg_conv_weight,bias=None)
        
        return pos_result-neg_result