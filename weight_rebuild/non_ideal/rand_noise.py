
import torch
import torch.nn as nn
import torch.nn.functional as F
from weight_rebuild.weight_mapping.weight_reshape import weight_to_rram
from weight_rebuild.weight_mapping.weight_reshape import rram_to_weight
import configparser as cp
import os

configs = cp.ConfigParser()
config_path = os.path.join(os.getcwd(), 'config.ini')
configs.read(config_path)

frequency = configs.getfloat('DEFAULT', 'frequency')
min_g = configs.getfloat('DEFAULT', 'min_g')
max_g = configs.getfloat('DEFAULT', 'max_g')
temperature = configs.getfloat('DEFAULT', 'temperature')
vdd = configs.getfloat('DEFAULT', 'vdd')
a = configs.getfloat('DEFAULT', 'a')
b = configs.getfloat('DEFAULT', 'b')
program_g = configs.getfloat('DEFAULT', 'program_g')

# cosntants
kb=1.380649e-23
q=1.602176634e-19

def rand_noise_gen(
    weight,
    min_w,
    max_w,
    en_thermal_noise=False,
    en_shot_noise=False,
    en_rtn=False,
    en_prog_error=False,
    temperature=temperature,
    frequency=frequency,
    vdd=vdd,
    cxb_shape=(64,64),
    device=torch.device('cuda:0')):

    rand_noise=weight_to_rram(weight,cxb_shape)

    rand_noise=(rand_noise-min_w)*(min_g-max_g)/(max_w-min_w)+max_g
    
    rtn_noise = torch.zeros(rand_noise.shape).to(device)
    normal_noise = torch.zeros(rand_noise.shape).to(device)
    
    if en_rtn:
        rtn_noise = -(a+b*rand_noise)*rand_noise/(a+rand_noise+b*rand_noise)

    if en_thermal_noise:
        normal_noise += 4*kb*temperature*frequency*rand_noise/((vdd/2)**2)
    if en_shot_noise:
        normal_noise += 2*q*vdd*frequency*rand_noise/((vdd/2)**2)
    if en_prog_error:
        normal_noise += program_g/3

    rand_noise=torch.normal(0,abs(torch.sqrt(normal_noise))) + rtn_noise
    
    rand_noise = (rand_noise - max_g) * (max_w - min_w) / (min_g - max_g) + min_w

    rand_noise=rram_to_weight(rand_noise,weight.shape,cxb_shape)
    
    return rand_noise
