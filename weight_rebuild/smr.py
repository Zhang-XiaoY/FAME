import copy
import torch
import torch.nn as nn

from weight_rebuild.cxb_conv2d import cxb_conv2d
from weight_rebuild.cxb_linear import cxb_linear

def smr(ori_model,
        cxb_shape,
        min_g,
        max_g,
        wire_g,
        SA0_ratio,
        SA1_ratio,
        device,
        en_nonidealities=[0,0,0,0,0]
        ):

    model = copy.deepcopy(ori_model)

    for name, layer in list(model.named_modules()):
        if isinstance(layer, nn.Conv2d):
            new_layer = cxb_conv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                weight=layer.weight,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=True,
                device=device,
                cxb_shape=cxb_shape,
                min_g=min_g,
                max_g=max_g,
                wire_g=wire_g,
                SA0_ratio=SA0_ratio,
                SA1_ratio=SA1_ratio,
                en_nonidealities=en_nonidealities
            )
            _set_module(model, name, new_layer)

        elif isinstance(layer, nn.Linear):
            # 替换linear层
            new_layer = cxb_linear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                weight=layer.weight,
                bias=True,
                device=device,
                cxb_shape=cxb_shape,
                min_g=min_g,
                max_g=max_g,
                wire_g=wire_g,
                SA0_ratio=SA0_ratio,
                SA1_ratio=SA1_ratio,
                en_nonidealities=en_nonidealities
            )
            new_layer=new_layer.to(device)
            _set_module(model, name, new_layer)

    return model


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_model = model
    for s in sub_tokens:
        cur_model = getattr(cur_model, s)
    setattr(cur_model, tokens[-1], module)
