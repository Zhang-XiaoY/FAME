
import torch
import torch.nn as nn
from weight_rebuild.weight_mapping.weight_reshape import rram_to_weight
from weight_rebuild.weight_mapping.weight_reshape import weight_to_rram

def SAF_mask_gen(weight_shape,weight_mapped_shape,SA0_ratio,SA1_ratio,device):

    pos_mask=torch.rand(weight_mapped_shape).to(device)
    neg_mask=torch.rand(weight_mapped_shape).to(device)

    pos_SA0_mask=torch.le(pos_mask,SA0_ratio).int()
    pos_SA1_mask=torch.ge(pos_mask,(1-SA1_ratio)).int()
    neg_SA0_mask=torch.le(neg_mask,SA0_ratio).int()
    neg_SA1_mask=torch.ge(neg_mask,(1-SA1_ratio)).int()
    cxb_shape=(weight_mapped_shape[2],weight_mapped_shape[3])

    pos_SA0_mask=rram_to_weight(pos_SA0_mask,weight_shape,cxb_shape)
    pos_SA1_mask=rram_to_weight(pos_SA1_mask,weight_shape,cxb_shape)
    neg_SA0_mask=rram_to_weight(neg_SA0_mask,weight_shape,cxb_shape)
    neg_SA1_mask=rram_to_weight(neg_SA1_mask,weight_shape,cxb_shape)
    
    return pos_SA0_mask,pos_SA1_mask,neg_SA0_mask,neg_SA1_mask


