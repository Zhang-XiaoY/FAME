import math
import torch.nn.functional as F

def cxb_count(data_size, cxb_shape):
    cxb_num=math.ceil(data_size/cxb_shape)
    pad_num=cxb_num*cxb_shape-data_size
    return cxb_num,pad_num


def cxb_count_2d(ori_weight_shape,cxb_shape):

    if len(ori_weight_shape)==4:
        flatten_shape=(ori_weight_shape[1]*ori_weight_shape[2]*ori_weight_shape[3],ori_weight_shape[0])
    elif len(ori_weight_shape)==2:
        flatten_shape=(ori_weight_shape[1],ori_weight_shape[0])

    cxb_row,pad_row=cxb_count(flatten_shape[0],cxb_shape[0])
    cxb_col,pad_col=cxb_count(flatten_shape[1],cxb_shape[1])
    weight_mapped_shape=(cxb_row,cxb_shape[0],cxb_col,cxb_shape[1])
    pad_shape=(pad_row,pad_col)
    return weight_mapped_shape,pad_shape

def weight_pad_del(weight,pad_shape):
    weight=weight[:weight.shape[0]-pad_shape[0],:weight.shape[1]-pad_shape[1]]
    return weight


def weight_to_rram(weight,cxb_shape):

    if len(weight.shape)==4:
        mapped_weight=weight.view(weight.shape[0],-1).transpose(1,0)
    elif len(weight.shape)==2:
        mapped_weight=weight.transpose(1,0)
    else:
        raise ValueError('weight shape error')

    weight_mapped_shape,pad_shape=cxb_count_2d(weight.shape,cxb_shape)
    mapped_weight=F.pad(mapped_weight,(0,pad_shape[1],0,pad_shape[0]))
    mapped_weight=mapped_weight.view(weight_mapped_shape).permute(0,2,1,3)
    return mapped_weight


def rram_to_weight(mapped_weight,ori_weight_shape,cxb_shape):

    flatten_shape,pad_shape=cxb_count_2d(ori_weight_shape,cxb_shape)
    mapped_weight_copy=mapped_weight.detach()
    weight=mapped_weight_copy.permute(0,2,1,3).contiguous().\
        view(flatten_shape[0]*flatten_shape[1],flatten_shape[2]*flatten_shape[3])

    weight=weight_pad_del(weight,pad_shape)

    if len(ori_weight_shape)==4:
        weight=weight.transpose(1,0).view(ori_weight_shape)
    elif len(ori_weight_shape)==2:
        weight=weight.transpose(1,0)
        
    return weight 
