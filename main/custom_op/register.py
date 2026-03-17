import logging
from functools import reduce

from .conv2d.conv_ASI import wrap_convASI
from .conv2d.conv_WSI import wrap_convWSI



from .linear.linear_ASI import wrap_linearASI
from .linear.linear_WSI import wrap_linearWSI
from .linear.linear_WASI import wrap_linearWASI


from .linear.linear_lora import wrap_linearLora
from .conv2d.conv_normal import wrap_conv

from .conv2d.conv_measure_perplexity_HOSVD import wrap_conv_measure_perplexity_HOSVD
from .linear.linear_measure_perplexity_HOSVD import wrap_linear_measure_perplexity_HOSVD


def register_ASI(module, cfgs):
    logging.info("Registering ASI")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        if module.model_type == "cnn":
            upd_layer = wrap_convASI(target, True, cfgs.get("activation_ranks", None)[layer_idx])
        elif module.model_type == "transformer":
            upd_layer = wrap_linearASI(target, True, cfgs["activation_ranks"][layer_idx], cfgs.get("truncation_threshold", None))


        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_normal_conv(module, cfgs):
    logging.info("Registering normal convolution")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["finetuned_layer"]:
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        if module.model_type == "cnn":
            upd_layer = wrap_conv(target, True)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_measure_perplexity_HOSVD(module, cfgs):
    logging.info("Measuring perplexity HOSVD")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')

        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        if module.model_type == "cnn":
            upd_layer = wrap_conv_measure_perplexity_HOSVD(target, True, cfgs["explain_variance_threshold"], cfgs["perplexity"], cfgs["measured_rank"], cfgs["layer_mem"], layer_idx)
        
        elif module.model_type == "transformer":
            upd_layer = wrap_linear_measure_perplexity_HOSVD(target, True, cfgs["explain_variance_threshold"], cfgs["perplexity"], cfgs["measured_rank"], cfgs["layer_mem"], layer_idx)


        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

from .linear.attn_vanilla import wrap_attn_vanilla
def register_attn_vanilla(module, attn_layers):
    logging.info("Registering Attn Vanilla")
    # Install filter
    for layer_idx, name in enumerate(attn_layers):
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        upd_layer = wrap_attn_vanilla(target)

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_WSI(module, cfgs):
    logging.info("Registering WSI")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        if module.model_type == "cnn":
            upd_layer = wrap_convWSI(target, True, cfgs["explained_variance_threshold"], layer_idx=layer_idx)
        elif module.model_type == "transformer":
            upd_layer = wrap_linearWSI(target, cfgs["explained_variance_threshold"], size=cfgs["size"], layer_idx=layer_idx, WSI_with_sub_iter=cfgs["WSI_with_sub_iter"])

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_WASI(module, cfgs):
    logging.info("Registering WASI")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False
        
        upd_layer = wrap_linearWASI(target, cfgs["activation_ranks"][layer_idx], cfgs["explained_variance_threshold"])

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_lora(module, cfgs):
    logging.info("Registering LORA")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        upd_layer = wrap_linearLora(target, 16, cfgs["rank"])
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)
