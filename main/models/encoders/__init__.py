from .resnet import resnet_encoders
from .mobilenet import mobilenet_encoders
from .mcunet import mcunet_encoders

from torchvision.models import swin_t, Swin_T_Weights, vit_b_32, ViT_B_32_Weights
import torch
from torchvision.models import (
    resnet34,
    resnet18,
    resnet50,
    mobilenet_v2,
    ResNet18_Weights,
    ResNet50_Weights,
    MobileNet_V2_Weights,
    ResNet34_Weights
)

encoders = {}
encoders.update(resnet_encoders)
encoders.update(mobilenet_encoders)
encoders.update(mcunet_encoders)


def get_encoder(name, checkpoint=None, is_pretrained=False, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):

    if not is_pretrained:
        if name == "resnet18": return resnet18(weights=ResNet18_Weights.DEFAULT)
        elif name == "resnet34": return resnet34(weights=ResNet34_Weights.DEFAULT)
        elif name == "mobilenet_v2": return mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        elif name == "resnet50": return resnet50(weights=ResNet50_Weights.DEFAULT)
        elif name == "swinT": return swin_t(weights=Swin_T_Weights.DEFAULT)
        elif name == "vit_b_32":
            if checkpoint is not None:
                pruned_dict = torch.load(checkpoint, weights_only=False, map_location='cpu') # Case SVD-LLM
                model = pruned_dict['model']
                return model
            elif weights == "full_imagenet":
                return vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
            elif weights == "raw":
                return vit_b_32(weights=None)
    else: # Raw checkpoint
        if name == "resnet18": return resnet18()
        elif name == "resnet34": return resnet34()
        elif name == "mobilenet_v2": return mobilenet_v2()
        elif name == "resnet50": return resnet50()
        elif name == "swinT": return swin_t()
        elif name == "vit_b_32": return vit_b_32()
    
    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(
            name, list(encoders.keys())))
    params = encoders[name]["params"]
    params.update(depth=depth)
    params.update(log_grad='log_grad' in kwargs and kwargs['log_grad'])
    encoder = Encoder(**params)
  
        
    if "mcunet" in name:
        assert "pretrained" in kwargs, "[Warning] pretrained condition is not defined for mcunet"
        encoder.set_in_channels(in_channels, pretrained=kwargs["pretrained"])
    # else:
    #     encoder.set_in_channels(in_channels, pretrained=weights is not None)
    
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder