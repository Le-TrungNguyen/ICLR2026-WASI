import logging
from functools import reduce

from .linear.linear import wrap_linear
from .linear.linear_WASI import wrap_linearWASI
from .linear.linear_lora import wrap_linearLora
from .linear.linear_ASI import wrap_linearASI


##########################################################################################################################

def register_normal_linear(module, cfgs):
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for name in cfgs["finetuned_layer"]:
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)
        if cfgs["type"] == "linear":
            upd_layer = wrap_linear(target, cfgs["backward_time"], cfgs["forward_time"], cfgs["inference_time"], energy_logger=cfgs["energy_logger"])

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)


def register_WASI(module, cfgs):
    logging.info("Registering WASI budget filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        upd_layer = wrap_linearWASI(linear=target, activation_ranks=cfgs["activation_ranks"][layer_idx], explained_variance_threshold=cfgs["explained_variance_threshold"], backward_time=cfgs["backward_time"], forward_time=cfgs["forward_time"], inference_time=cfgs["inference_time"], energy_logger=cfgs["energy_logger"], output_calculation_time = cfgs["output_calculation_time"], orthogonalization_time = cfgs["orthogonalization_time"], matmuls_time = cfgs["matmuls_time"])

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_ASI(module, cfgs):
    logging.info("Registering ASI budget filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False


        upd_layer = wrap_linearASI(linear=target, active=True, rank=cfgs["activation_ranks"][layer_idx], backward_time=cfgs["backward_time"], forward_time=cfgs["forward_time"], inference_time=cfgs["inference_time"])

        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)

def register_lora(module, cfgs):
    logging.info("Registering LORA filter")
    if cfgs == -1:
        logging.info("No Filter Required")
        return
    # Install filter
    for layer_idx, name in enumerate(cfgs["finetuned_layer"]):
        path_seq = name.split('.')
        target = reduce(getattr, path_seq, module)

        for param in target.parameters(): # Turn off gradient of previous version
            param.requires_grad = False

        upd_layer = wrap_linearLora(target, 16, cfgs["rank"], backward_time=cfgs["backward_time"], forward_time=cfgs["forward_time"], inference_time=cfgs["inference_time"])
        parent = reduce(getattr, path_seq[:-1], module)
        setattr(parent, path_seq[-1], upd_layer)