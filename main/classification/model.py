import os
import numpy as np
import torch as th
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from custom_op.register import register_ASI, register_measure_perplexity_HOSVD, register_WSI, register_WASI, register_lora, register_attn_vanilla
from utils.util import get_all_layer_with_name, get_active_layer_with_name, Hook, calculate_flops_subspace_iteration, calculate_flops_SVD, get_all_attn_with_name
from models.encoders import get_encoder
from functools import reduce
import logging
import inspect
from utils.perplexity_dp import Perplexity

from custom_op.linear.linear_lora import LoRALinear
from custom_op.linear.linear_ASI import Linear_ASI
from custom_op.linear.linear_WSI import Linear_WSI
from custom_op.linear.linear_WASI import Linear_WASI, SVD_var
from custom_op.conv2d.conv_ASI import Conv2d_ASI
from custom_op.conv2d.conv_WSI import Conv2d_WSI

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ClassificationModel(LightningModule):
    def __init__(self, backbone: str, backbone_args, num_classes,
                 learning_rate, weight_decay, set_bn_eval, load = None,

                 num_of_finetune=None,
                
                # ASI params:
                 measure_perplexity_HOSVD_var=False,
                 with_ASI=False,
                 ASI_threshold=False, 
                 ASI_rank=None,
                 budget=None,
                 perplexity_pkl=None,

                # WASI params:
                 with_WSI=False, 
                 with_WASI=False, 

                # WSI params:
                 WSI_with_sub_iter=True,
                 
                # LoRA params:
                 with_lora=False, 
                 lora_alpha=None,
                 truncation_threshold=None,   
                 
                # Other params:
                 count_attention = False, 
                 is_pretrained=False, 
                 just_log = False,  
                 checkpoint=None,

                 use_sgd=False, momentum=0.9, anneling_steps=8008, scheduler_interval='step',
                 lr_warmup=0, init_lr_prod=0.25):
                
        super(ClassificationModel, self).__init__()

        # Automatically capture all init arguments and their values
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self.initial_state = {arg: values[arg] for arg in args if arg != 'self'}
        self.checkpoint = checkpoint
        ############################################################
        self.num_classes = num_classes

        # Get model
        self.backbone_name = backbone
        self.tokenizer = None
        if self.backbone_name == "TinyLlama":
            self.do_on_layer = True
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Kiểm tra và thêm pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # Dùng </s> làm pad_token
                print(f"Set pad_token to: {self.tokenizer.pad_token}")
            self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_classes)
            self.backbone.config.pad_token_id = self.tokenizer.pad_token_id  # Đồng bộ pad_token_id
            print(f"Model pad_token_id: {self.backbone.config.pad_token_id}")
        else:
            self.backbone = get_encoder(backbone, checkpoint, is_pretrained, **backbone_args)

        ### change the classifier head ###
        if self.backbone_name == "swinT":
            self.backbone.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        elif self.backbone_name == "vit_b_32":
            self.backbone.heads = nn.Sequential(nn.Linear(in_features=768, out_features=num_classes, bias=True))
        elif self.backbone_name in ["resnet18", "resnet34", "resnet50"]:
            self.backbone.fc = nn.Linear(in_features=self.backbone.fc.in_features, out_features=num_classes, bias=True)
        elif self.backbone_name == "mobilenet_v2":
            self.backbone.classifier[1] = nn.Linear(in_features=self.backbone.classifier[1].in_features, out_features=num_classes, bias=True)
        elif self.backbone_name == "mcunet":
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(self.backbone._out_channels[-1], num_classes)

        if self.backbone_name in ["swinT", "vit_b_32", "TinyLlama"]: self.model_type = "transformer"
        elif self.backbone_name in ["resnet18", "resnet34", "resnet50", "mobilenet_v2", "mcunet"]: self.model_type = "cnn"
        

        #############################################################
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.set_bn_eval = set_bn_eval
        self.acc = Accuracy(num_classes=num_classes)
        self.is_pretrained = is_pretrained
        self.just_log = just_log
        #############################################################

        self.count_attention = count_attention
        if self.count_attention:
            self.all_attns = get_all_attn_with_name(self)
            register_attn_vanilla(self, self.all_attns)

        ###############################################################

        self.all_layers = get_all_layer_with_name(self)

        if num_of_finetune == "all" or num_of_finetune > len(self.all_layers):
            logging.info("[Warning] Finetuning all layers")
            self.finetune_all = True
            self.num_of_finetune = len(self.all_layers)
        else:
            self.num_of_finetune = num_of_finetune
            self.finetune_all = False

        ###################################################################

        self.truncation_threshold = truncation_threshold

        ## For ASI
        self.with_ASI = with_ASI
        self.ASI_threshold = ASI_threshold
        self.ASI_rank = ASI_rank
        self.measure_perplexity_HOSVD_var = measure_perplexity_HOSVD_var

        if self.measure_perplexity_HOSVD_var:
            self.perplexity     = [None for layer_idx in range(len(self.all_layers))]
            self.measured_rank  = [None for layer_idx in range(len(self.all_layers))]
            self.layer_mem      = [None for layer_idx in range(len(self.all_layers))]

        
        

        ## For WSI and WASI
        self.with_WSI = with_WSI
        self.WSI_with_sub_iter = WSI_with_sub_iter

        self.with_WASI = with_WASI

        if self.with_ASI or self.with_WASI:
            if self.backbone_name == "TinyLlama":
                self.suitable_ranks = [self.truncation_threshold for layer_idx in range(self.num_of_finetune)]
            else:
                self.perplexity_pkl = perplexity_pkl
                perplexity = Perplexity()
                perplexity.load(self.perplexity_pkl)
                best_memory, best_perplexity, best_indices, self.suitable_ranks = perplexity.find_best_ranks_dp(budget=budget, num_of_finetuned=self.num_of_finetune)
                del perplexity

        ## For LoRA
        self.with_lora = with_lora
        self.lora_alpha = lora_alpha

        self.with_base = not (self.with_ASI or self.with_WASI or self.with_lora or self.with_WSI)

        
        ########################## 
        self.use_sgd = use_sgd
        self.momentum = momentum
        self.anneling_steps = anneling_steps
        self.scheduler_interval = scheduler_interval
        self.lr_warmup = lr_warmup
        self.init_lr_prod = init_lr_prod
        self.hook = {} # Hook being a dict: where key is the module name and value is the hook

        ###################################### Create configuration to modify model #########################################
        self.filter_cfgs = {"backbone": backbone}
        self.handle_finetune()
        #######################################################################
        

        if load != None:
            state_dict = th.load(load)['state_dict']
            self.load_state_dict(state_dict)
        
        if self.measure_perplexity_HOSVD_var:
            register_measure_perplexity_HOSVD(self, self.filter_cfgs)
        elif self.with_ASI:
            register_ASI(self, self.filter_cfgs)
        elif self.with_WSI:
            register_WSI(self, self.filter_cfgs)
        elif self.with_WASI:
            register_WASI(self, self.filter_cfgs)
        elif self.with_lora:
            register_lora(self, self.filter_cfgs)

        self.acc.reset()


    def attach_memory_list_weight(self):
        self.US_shape1 = []
        self.k = []
        self.VT_shape2 = []
        self.input_shapes = []
        self.non_zeros_in_sparse = []
        if not self.WSI_with_sub_iter:
            self.layer_eigen_values = [[] for i in range(self.num_of_finetune)]
            self.size = [self.US_shape1, self.k, self.VT_shape2, self.input_shapes, self.layer_eigen_values, self.non_zeros_in_sparse]
        else:
            self.size = [self.US_shape1, self.k, self.VT_shape2, self.input_shapes, self.non_zeros_in_sparse]

        self.filter_cfgs["size"] = self.size

        if self.with_WSI:
            register_WSI(self, self.filter_cfgs)

        self.update_optimizer()
  
    def reset_memory_list_weight(self):
        self.US_shape1.clear()
        self.k.clear()
        self.VT_shape2.clear()
        self.input_shapes.clear()
        self.non_zeros_in_sparse.clear()
        if not self.WSI_with_sub_iter:
            for eigen_value in self.layer_eigen_values:
                eigen_value.clear()

    def clear_measured_variables(self):
        for i in range(len(self.perplexity)):
            self.perplexity[i]     = None
            self.measured_rank[i]  = None
            self.layer_mem[i]      = None
    
    def reset(self):
        # Reset the model to its initial state
        self.__init__(**self.initial_state)


    def set_filter_configs(self, finetuned_layer):
        """ Helper function to set filter configurations based on conditions """
        if finetuned_layer == None: self.filter_cfgs = -1
        else:
            new_items = {}
            self.filter_cfgs["finetuned_layer"] = finetuned_layer
            if self.measure_perplexity_HOSVD_var:
                new_items = {"explain_variance_threshold": self.truncation_threshold, "perplexity": self.perplexity, "measured_rank": self.measured_rank, "layer_mem": self.layer_mem}

            elif self.with_ASI:
                if self.ASI_rank is not None: # Truncate based on rank
                    new_items = {"activation_ranks": [self.ASI_rank for i in range(self.num_of_finetune)]}
                elif not self.ASI_threshold: # Truncate based on perplexity
                    new_items = {"activation_ranks": self.suitable_ranks}
                else: # Truncated based on explained variance threshold
                    new_items = {"activation_ranks": self.suitable_ranks, "truncation_threshold": self.truncation_threshold}
            elif self.with_WSI:
                new_items = {"explained_variance_threshold": self.truncation_threshold, "size": None, "WSI_with_sub_iter": self.WSI_with_sub_iter}
            elif self.with_WASI:
                new_items = {"activation_ranks": self.suitable_ranks, "explained_variance_threshold": self.truncation_threshold}
            elif self.with_lora:
                new_items = {"lora_alpha": self.lora_alpha, "rank": self.truncation_threshold}
            
            self.filter_cfgs.update(new_items)
    
    def freeze_layers(self):
        """ Helper function to freeze layers that are not being finetuned """
        if self.num_of_finetune != 0 and self.num_of_finetune != None and not self.is_pretrained:
            if self.num_of_finetune != "all" and self.num_of_finetune <= len(self.all_layers):
                self.all_layers = dict(list(self.all_layers.items())[-self.num_of_finetune:])

            for name, mod in self.named_modules():
                if len(list(mod.children())) == 0:
                    if name not in self.all_layers and name != '':
                        mod.eval()
                        for param in mod.parameters():
                            param.requires_grad = False  # Freeze layer
                    elif name in self.all_layers:
                        break
            return self.all_layers
        elif self.is_pretrained:
            self.all_layers = dict(list(self.all_layers.items())[-self.num_of_finetune:])
            return self.all_layers
        else:
            for name, mod in self.named_modules():
                if len(list(mod.children())) == 0:
                    if name != '':
                        mod.eval()
                        for param in mod.parameters():
                            param.requires_grad = False  # Freeze layer
            return None
    
    def handle_finetune(self):
        if not self.measure_perplexity_HOSVD_var:
            """ Handle the logic for finetuning based on num_of_finetune """
            if self.num_of_finetune != 0 and self.num_of_finetune != "all" and self.finetune_all == False:
                self.all_layers = self.freeze_layers()
            elif self.num_of_finetune == 0 or self.num_of_finetune == None: # If no finetune => freeze all
                logging.info("[Warning] number of finetuned layers is 0 => Freeze all layers !!")
                self.all_layers = self.freeze_layers()
            elif self.finetune_all:
                logging.info("[Warning] Finetune all model!!")
            else:
                logging.info("[Warning] Missing configuration !!")
                self.all_layers = None
            self.set_filter_configs(self.all_layers)
        else:
            self.set_filter_configs(self.all_layers)

    def activate_hooks(self, hook, is_activated=True):
        for h in hook:
            if isinstance(hook[h], dict):
                for sub in hook[h].values():
                    sub.activate(is_activated)
            else:
                hook[h].activate(is_activated)
        logging.info(f"Hooks {'activated' if is_activated else 'deactivated'}")


    def remove_hooks(self, hook):
        for h in hook:
            if isinstance(hook[h], dict):
                for sub in hook[h].values():
                    sub.remove()
            else:
                hook[h].remove()
        hook.clear()
        logging.info("Hook is removed")

    def attach_hooks(self, consider_active_only=False):
        if not consider_active_only:
            layers = get_all_layer_with_name(self)
        else:
            layers = get_active_layer_with_name(self)
        assert layers != -1, f"[Warning] Consider activate {self.model_type} layers only but no one is finetuned => No hook is attached !!"

        for name, mod in  layers.items():
            self.hook[name] = Hook(mod)


    def get_resource_consumption(self, consider_active_only=False, element_size=4, unit="MB", register_hook=False): # element_size = 4 bytes
        # Register hook to log input/output size
        if register_hook:
            self.attach_hooks(consider_active_only=consider_active_only)
            self.activate_hooks(self.hook, True)
        #############################################################################
        else:
            _, first_hook = next(iter(self.hook.items()))
            if first_hook.active: logging.info("Hook is activated")
            else: logging.info("[Warning] Hook is not activated !!")
            #############################################################################
            # Feed one sample of data into model to record activation size
            num_element_activation = 0
            self.num_flops_fw = 0
            num_flops_inference = 0
            num_flops_bw = 0
            self.num_element_weight = 0
            num_element_weight_finetune = 0

            for layer_index, name in enumerate(self.hook): # through each layer
                if self.model_type == "transformer":
                    input_size = self.hook[name].input_size

                    if input_size is None: continue # Maybe attention doesn't receive input, so skip
                    elif input_size.numel() == 4: # Case: SwinT
                        B, H, W, C = [x.item() for x in input_size]
                        _, _, _, C_prime  = [x.item() for x in self.hook[name].output_size]

                        weight_size = th.tensor(self.hook[name].weight_size)

                        if layer_index < len(self.hook) - self.num_of_finetune:
                            self.num_element_weight += int(weight_size.prod())
                            
                            self.num_flops_fw += B * H * 2 * W * C * C_prime
                            num_flops_inference += B * H * 2 * W * C * C_prime

                        else:
                            if isinstance(self.hook[name].module, Linear_WSI) and self.WSI_with_sub_iter and self.with_WSI:

                                # Calculate activation size
                                num_element_activation += int(input_size.prod())

                                # Calculate weight size
                                # Uk, Sk, Vtk, K_weight = SVD_var(self.hook[name].weight, self.truncation_threshold)
                                # num_element_weight_finetune += Uk.numel() + Vtk.numel()
                                L = self.hook[name].L_weight
                                R = self.hook[name].R_weight
                                num_element_weight_finetune += L.numel() + R.numel()
                                if L is None or R is None:
                                    continue

                                # rank
                                K_weight = R.shape[0]   # or L.shape[1]
                                
                                # Calculate FLOPs
                                # FLOPs of compressing weight:
                                fw_overhead_weight = calculate_flops_subspace_iteration(weight_size[0].item(), weight_size[1].item(), K_weight)

                                # FLOPs of performing low rank forward
                                low_rank_fw = B*H*W*K_weight*(2*C - 1) + B*H*W*C_prime*(2*K_weight - 1)
                                
                                # FLOPs of performing backward to calculate weight gradient
                                bw_grad_weight = int(C_prime*C*(2*B*H*W - 1))

                                # FLOPs of performing backward to calculate activation gradient
                                bw_grad_activation = B*H*2*W*K_weight*(C_prime + C)

                                self.num_flops_fw += fw_overhead_weight + low_rank_fw
                                num_flops_inference += low_rank_fw
                                num_flops_bw += bw_grad_weight + bw_grad_activation


                            elif isinstance(self.hook[name].module, LoRALinear):

                                # Calculate activation size
                                num_element_activation += int(input_size.prod())

                                # Calculate weight size
                                num_element_weight_finetune += int(weight_size.prod())
                                
                                # Calculate LoRA adapter weight size
                                num_element_weight_finetune += int(self.hook[name].lora_A_weight.numel()) + int(self.hook[name].lora_B_weight.numel())

                                # Get LORA rank
                                R = self.hook[name].lora_rank

                                ############## Calculate FLOPs
                                num_flops_inference += int(B*H*C_prime*W*(2*C-1))

                                #### Forward pass flops
                                # Forward through loraA: BHWC and CR -> BHWR
                                lora_fw_A = int(B * H * W * R * (2*C - 1))

                                # Forward through loraB: BHWR and RC' -> BHWC'
                                lora_fw_B = int(B * H * W * C_prime * (2*R - 1))

                                # Original forward: Y = XW^T
                                original_fw = int(B * H * W * C_prime * (2*C - 1))

                                # Multiply with scaling factor and add to original output
                                lora_fw_scaling = int(B * H * W * C_prime * 2)  # Multiply with scaling factor and add

                                # Total FLOPs forward
                                fw_flops = original_fw + lora_fw_A + lora_fw_B + lora_fw_scaling

                                #### Backward pass flops
                                # LoRA B: 
                                # BHWC' and BHWR -> RC'
                                bw_grad_weight_B = int(R * C_prime * (2 * B * H * W - 1))
                                # BHWC' and RC' -> BHWR
                                bw_grad_input_B = int(B * H * W * R * (2 * C_prime - 1))

                                # LoRA A:
                                # BHWR and BHWC -> RC
                                bw_grad_weight_A = int(R * C * (2 * B * H * W - 1))
                                # BHWR and RC -> BHWC
                                bw_grad_input_A = int(B * H * W * C * (2 * R - 1))

                                bw_flops = bw_grad_weight_A + bw_grad_weight_B + bw_grad_input_A + bw_grad_input_B

                                self.num_flops_fw += fw_flops
                                num_flops_bw += bw_flops



                            elif isinstance(self.hook[name].module, Linear_ASI):
                                # Calculate activation size
                                from custom_op.compression.rank.hosvd_power_iteration import hosvd_power_iteration
                                S, u_list = hosvd_power_iteration(self.hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=self.suitable_ranks[layer_index - (len(self.hook) - self.num_of_finetune)]) # Layer index là index của layer trong model, không phải trong suitable rank (chỉ xét các finetune layer), vì vậy cần điều chỉnh
                                num_element_activation += S.numel() + sum(u.numel() for u in u_list)

                                # Calculate weight size
                                num_element_weight_finetune += int(weight_size.prod())

                                # Calculate FLOPs
                                K1, K2, K3, K4 = S.shape

                                # Forward = overhead của ASI + vanilla cost
                                fw_overhead = calculate_flops_subspace_iteration(B, C*H*W, K1)
                                fw_overhead += calculate_flops_subspace_iteration(C, B*H*W, K2)
                                fw_overhead += calculate_flops_subspace_iteration(H, B*C*W, K3)
                                fw_overhead += calculate_flops_subspace_iteration(W, B*C*H, K4)

                                vanilla_fw = (B*H*C_prime*W*(2*C-1))
                                

                                # Backward = cost to calculate gradient for weight in low rank
                                bw = (B*K1*H*W*C_prime + H*K2*K1*K3*K4 + W*K3*K1*H*C_prime + C*K4*K1*H*K3 + K1*K3*H*C*C_prime)

                                # Add cost to calculate gradient for activation
                                bw += int(B*H*W*C*(2*C_prime-1))

                                self.num_flops_fw += fw_overhead + vanilla_fw
                                num_flops_bw += bw

                            elif isinstance(self.hook[name].module, Linear_WASI): # Calculate size of weight and activation if using WASI
                                # This method fixes memory for weight and activation
                                # Calculate memory for activation
                                from custom_op.compression.rank.hosvd_power_iteration import hosvd_power_iteration
                                S_activation, u_list = hosvd_power_iteration(self.hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=self.suitable_ranks[layer_index - (len(self.hook) - self.num_of_finetune)])
                                num_element_activation += S_activation.numel() + sum(u.numel() for u in u_list)

                                # Calculate memory for weight
                                # Uk, Sk, Vtk, K_weight = SVD_var(self.hook[name].weight, self.truncation_threshold)
                                # num_element_weight_finetune += Uk.numel() + Vtk.numel()
                                # Read L and R directly from hook
                                L = self.hook[name].L_weight
                                R = self.hook[name].R_weight

                                if L is None or R is None:
                                    continue
                                num_element_weight_finetune += L.numel() + R.numel()
                                # rank
                                K_weight = R.shape[0]   # or L.shape[1]



                                # Calculate FLOPs
                                K1, K2, K3, K4 = S_activation.shape

                                # FLOPs to compress activation:
                                fw_overhead_activation = calculate_flops_subspace_iteration(B, C*H*W, K1)
                                fw_overhead_activation += calculate_flops_subspace_iteration(C, B*H*W, K2)
                                fw_overhead_activation += calculate_flops_subspace_iteration(H, B*C*W, K3)
                                fw_overhead_activation += calculate_flops_subspace_iteration(W, B*C*H, K4)

                                # FLOPs to compress weight:
                                fw_overhead_weight = calculate_flops_subspace_iteration(weight_size[0].item(), weight_size[1].item(), K_weight)

                                # FLOPs to Forward with compressed weight
                                low_rank_fw = B*H*W*K_weight*(2*C - 1) + B*H*W*C_prime*(2*K_weight - 1)

                                # Backward to calculate gradient for weight
                                bw_grad_weight = (B*K1*H*W*C_prime + H*K2*K1*K3*K4 + W*K3*K1*H*C_prime + C*K4*K1*H*K3 + K1*K3*H*C*C_prime)

                                # Backward to calculate gradient for activation
                                bw_grad_activation = B*H*2*W*K_weight*(C_prime + C)

                                self.num_flops_fw += fw_overhead_activation + fw_overhead_weight + low_rank_fw # Compress activation, weight and low rank fw
                                num_flops_inference += low_rank_fw # Compress weight and low rank fw
                                num_flops_bw += bw_grad_weight + bw_grad_activation

                            elif isinstance(self.hook[name].module, nn.modules.linear.Linear):
                                num_element_activation += int(input_size.prod())

                                num_element_weight_finetune += int(weight_size.prod()) # Measure weight of baseline

                                vanilla_fw = int(B*H*C_prime*W*(2*C-1))

                                vanilla_bw_grad_weight = int(C_prime*C*(2*B*H*W - 1))
                                vanilla_bw_grad_input = int(B*H*C_prime*C*(2*W-1))

                                vanilla_bw = vanilla_bw_grad_weight + vanilla_bw_grad_input


                                self.num_flops_fw += vanilla_fw
                                num_flops_bw += vanilla_bw

                    elif input_size.numel() == 3: # Case: TinyLlama and ViT
                        B, N, I = [x.item() for x in input_size]
                        _, _, O  = [x.item() for x in self.hook[name].output_size]
                        weight_size = th.tensor(self.hook[name].weight_size)

                        if layer_index < len(self.hook) - self.num_of_finetune:
                            self.num_element_weight += int(weight_size.prod())
                            self.num_flops_fw += int(B*I*O*(2*N-1))
                            num_flops_inference += int(B*I*O*(2*N-1))

                        else:
                            if isinstance(self.hook[name].module, Linear_WSI) and self.WSI_with_sub_iter and self.with_WSI:
                                num_element_activation += int(input_size.prod())

                                # Uk, Sk, Vtk, K_weight = SVD_var(self.hook[name].weight, self.truncation_threshold)
                                # num_element_weight_finetune += Uk.numel() + Vtk.numel()

                                L = self.hook[name].L_weight
                                R = self.hook[name].R_weight
                                num_element_weight_finetune += L.numel() + R.numel()
                                if L is None or R is None:
                                    continue

                                # rank
                                K_weight = R.shape[0]   # or L.shape[1]


                                fw_overhead_weight = calculate_flops_subspace_iteration(weight_size[0].item(), weight_size[1].item(), K_weight)

                                low_rank_fw = B*N*K_weight*(2*I-1) + B*N*O*(2*K_weight - 1)

                                bw_grad_weight = int(O*I*(2*B*N-1))

                                bw_grad_activation = B*N*K_weight*(2*I-1) + B*N*O*(2*K_weight - 1)

                                self.num_flops_fw += fw_overhead_weight + low_rank_fw # Compress activation, weight and low rank fw
                                num_flops_inference += low_rank_fw # Compress weight and low rank fw
                                num_flops_bw += bw_grad_weight + bw_grad_activation

                            elif isinstance(self.hook[name].module, LoRALinear):
                                # Calculate activation size
                                num_element_activation += int(input_size.prod())

                                # Calculate weight size
                                num_element_weight_finetune += int(weight_size.prod()) # Measure weight of baseline

                                # Add weight for adapter
                                num_element_weight_finetune += int(self.hook[name].lora_A_weight.numel()) + int(self.hook[name].lora_B_weight.numel())

                                # Get rank of LoRA
                                R = self.hook[name].lora_rank  # Rank of LoRA

                                ############## Calculate FLOPs
                                num_flops_inference += int(B*N*O*(2*I-1))

                                #### Forward pass flops
                                # Forward through loraA: BNI and IR -> BNR
                                lora_fw_A = int(B * N * R * (2*I - 1))

                                # Forward through loraB: BNR and RO -> BNO
                                lora_fw_B = int(B * N * O * (2*R - 1))

                                # Original forward: Y = XW^T
                                original_fw = (B*N*O*(2*I-1))

                                # Multiply by scaling factor and add to original output
                                lora_fw_scaling = int(B * N * O * 2)
                                # Total FLOPs forward
                                fw_flops = original_fw + lora_fw_A + lora_fw_B + lora_fw_scaling

                                #### Backward pass flops
                                # LoRA B: 
                                # BNO and BNR -> RO
                                bw_grad_weight_B = int(R * O * (2 * B * N - 1))
                                # BNO and RO -> BNR
                                bw_grad_input_B = int(B * N * R * (2 * O - 1))

                                # LoRA A:
                                # BNR and BNI -> IR
                                bw_grad_weight_A = int(R * I * (2 * B * N - 1))
                                # BNR and IR -> BNI
                                bw_grad_input_A = int(B * N * I * (2 * R - 1))

                                bw_flops = bw_grad_weight_A + bw_grad_weight_B + bw_grad_input_A + bw_grad_input_B

                                self.num_flops_fw += fw_flops
                                num_flops_bw += bw_flops

                            elif isinstance(self.hook[name].module, Linear_ASI):
                                # Calculate activation size
                                from custom_op.compression.rank.hosvd_power_iteration import hosvd_power_iteration
                                S, u_list = hosvd_power_iteration(self.hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=self.suitable_ranks[layer_index - (len(self.hook) - self.num_of_finetune)])
                                num_element_activation += S.numel() + sum(u.numel() for u in u_list)

                                # Calculate weight size
                                num_element_weight_finetune += int(weight_size.prod())

                                ########################## Calculate FLOPs ######################
                                K1, K2, K3 = S.shape
                                
                                # Forward = overhead of ASI + vanilla cost
                                fw_overhead_activation = calculate_flops_subspace_iteration(B, N*I, K1)
                                fw_overhead_activation += calculate_flops_subspace_iteration(I, B*N, K2)
                                fw_overhead_activation += calculate_flops_subspace_iteration(N, B*I, K3)
                                vanilla_fw = int(B*N*O*(2*I-1))
                                # Backward = cost to calculate gradient for weight at low rank
                                bw = (B*N*O*K1 + K1*K2*K3*N + K1*K3*I*N + I*O*N*K1)

                                # Add cost to calculate gradient for activation
                                bw += int(B*N*I*(2*O - 1))

                                self.num_flops_fw += fw_overhead_activation + vanilla_fw
                                num_flops_bw += bw

                            elif isinstance(self.hook[name].module, Linear_WASI): # Calculate size of weight and activation if using WASI
                                # Calculate mem for activation
                                from custom_op.compression.rank.hosvd_power_iteration import hosvd_power_iteration
                                S_activation, u_list = hosvd_power_iteration(self.hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=self.suitable_ranks[layer_index - (len(self.hook) - self.num_of_finetune)])
                                num_element_activation += S_activation.numel() + sum(u.numel() for u in u_list)

                                # Calculate mem for weight
                                # Uk, Sk, Vtk, K_weight = SVD_var(self.hook[name].weight, self.truncation_threshold)
                                # num_element_weight_finetune += Uk.numel() + Vtk.numel()

                                L = self.hook[name].L_weight
                                R = self.hook[name].R_weight
                                num_element_weight_finetune += L.numel() + R.numel()
                                if L is None or R is None:
                                    continue

                                # rank
                                K_weight = R.shape[0]   # or L.shape[1]


                                # Calculate FLOPs
                                K1, K2, K3 = S_activation.shape

                                # FLOPs to compress activation:
                                fw_overhead_activation = calculate_flops_subspace_iteration(B, N*I, K1)
                                fw_overhead_activation += calculate_flops_subspace_iteration(I, B*N, K2)
                                fw_overhead_activation += calculate_flops_subspace_iteration(N, B*I, K3)

                                # FLOPs to compress weight:
                                fw_overhead_weight = calculate_flops_subspace_iteration(weight_size[0].item(), weight_size[1].item(), K_weight)

                                # FLOPs to Forward with compressed weight
                                low_rank_fw = B*N*K_weight*(2*I-1) + B*N*O*(2*K_weight - 1)

                                # Backward to calculate gradient for weight
                                bw_grad_weight = (B*N*O*K1 + K1*K2*K3*N + K1*K3*I*N + I*O*N*K1)

                                # Backward to calculate gradient for activation
                                bw_grad_activation = B*N*K_weight*(2*I-1) + B*N*O*(2*K_weight - 1)

                                self.num_flops_fw += fw_overhead_activation + fw_overhead_weight + low_rank_fw # Compress activation, weight and low rank fw
                                num_flops_inference += low_rank_fw # low rank fw
                                num_flops_bw += bw_grad_weight + bw_grad_activation

                            elif isinstance(self.hook[name].module, nn.modules.linear.Linear):
                                num_element_activation += int(input_size.prod())

                                num_element_weight_finetune += int(weight_size.prod()) # Measure weight of baseline

                                # Calculate FLOPs
                                vanilla_fw = int(B*N*O*(2*I-1))

                                vanilla_bw_grad_weight = int(O*I*(2*B*N-1))
                                vanilla_bw_grad_input = int(B*N*I*(2*O-1))
                                vanilla_bw = vanilla_bw_grad_weight + vanilla_bw_grad_input

                                self.num_flops_fw += vanilla_fw
                                num_flops_bw += vanilla_bw
            
                elif self.model_type == "cnn":
                    input_size = self.hook[name].input_size
                    if layer_index < len(self.hook) - self.num_of_finetune:
                        num_element_weight += th.prod(th.tensor(self.hook[name].weight.shape)).item()
                    
                    else:
                        if isinstance(self.hook[name].module, Conv2d_WSI):
                            from custom_op.compression.explain_var.hosvd_4_mode_var import hosvd_4_mode_var
                            S_weight, u_list_weight, previous_rank = hosvd_4_mode_var(self.hook[name].weight, self.truncation_threshold, True)
                            num_element_weight_finetune += S_weight.numel() + sum(u.numel() for u in u_list_weight)

                            num_element_activation += int(input_size[1] * input_size[2] * input_size[3] * input_size[0])
                            

                        elif isinstance(self.hook[name].module, Conv2d_ASI):
                            from custom_op.compression.rank.hosvd_power_iteration import hosvd_power_iteration
                            if not self.with_ASI:
                                S, u_list = hosvd_power_iteration(self.hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=self.truncation_threshold)
                            else:
                                S, u_list = hosvd_power_iteration(self.hook[name].inputs[0], previous_Ulist=None, reuse_U=False, rank=self.suitable_ranks[layer_index])

                            num_element_activation += S.numel() + sum(u.numel() for u in u_list)

                            num_element_weight_finetune += th.prod(th.tensor(self.hook[name].weight.shape)).item()

                        elif isinstance(self.hook[name].module, nn.modules.conv.Conv2d) and self.with_base:
                            num_element_activation += int(input_size[1] * input_size[2] * input_size[3] * input_size[0])
                            num_element_weight_finetune += th.prod(th.tensor(self.hook[name].weight.shape)).item()


            self.remove_hooks(self.hook)

            if unit == "Byte":
                res_activation = str(num_element_activation*element_size)
                res_weight_finetuned = str(num_element_weight_finetune*element_size)
                res_weight = str((self.num_element_weight + num_element_weight_finetune)*element_size)
            if unit == "MB":
                res_activation = str((num_element_activation*element_size)/(1024*1024))
                res_weight_finetuned = str(num_element_weight_finetune*element_size/(1024*1024))
                res_weight = str((self.num_element_weight + num_element_weight_finetune)*element_size/(1024*1024))
            elif unit == "KB":
                res_activation = str((num_element_activation*element_size)/(1024))
                res_weight_finetuned = str(num_element_weight_finetune*element_size/(1024))
                res_weight = str((self.num_element_weight + num_element_weight_finetune)*element_size/(1024))
            else:
                raise ValueError("Unit is not suitable")

            with open(os.path.join(self.logger.log_dir, f'activation_memory_{unit}.log'), "a") as file:
                file.write(f"Activation memory is {res_activation} {unit}\n")

            if self.WSI_with_sub_iter:
                with open(os.path.join(self.logger.log_dir, f'weight_memory_{unit}.log'), "a") as file:
                    file.write(f"Weight memory is {res_weight} {unit}\n")
                    file.write(f"Weight of finetuned layer is {res_weight_finetuned} {unit}\n")

                with open(os.path.join(self.logger.log_dir, f'total_FLOPs.log'), "a") as file:
                    file.write(f"Forward: {self.num_flops_fw}\n")
                    file.write(f"Backward: {num_flops_bw}\n")
                    file.write(f"Total FLOPs: {self.num_flops_fw + num_flops_bw}\n")
                    if self.with_WASI or self.with_lora or self.with_WSI:
                        file.write(f"Inference: {num_flops_inference}\n")

    def get_weight_size_WSI(self, num_batches, element_size=4, unit="MB"):
        if self.model_type != "transformer":
            raise ValueError("This function is only for transformer model with WSI method !!")
        
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        weight_size_tensor = th.tensor(self.size[:3], device=device).float() # Shape: (3 shape, #batches * num_of_finetune)
        weight_size_tensor = weight_size_tensor.view(3, num_batches, -1) # Shape: (3 shape, #batches, num_of_finetune)
        weight_size_tensor = weight_size_tensor.permute(2, 1, 0) # Shape: (num_of_finetune, #batch, 3 shape)

        non_zeros_in_sparse = th.tensor(self.size[-1], device=device).float()  # Shape: (1 shape, #batches * num_of_finetune)
        non_zeros_in_sparse = non_zeros_in_sparse.view(num_batches, -1)  # Shape: (1 shape, #batches, num_of_finetune)
        non_zeros_in_sparse = non_zeros_in_sparse.sum(dim=1)  # Shape: (#batches,)
        non_zeros_in_sparse = non_zeros_in_sparse.mean().item()  # Average over batches



        num_element_all = th.sum(
            weight_size_tensor[:, :, 0] * weight_size_tensor[:, :, 1] + weight_size_tensor[:, :, 1] * weight_size_tensor[:, :, 2],
            dim=1
        )
        num_element = th.sum(num_element_all) / weight_size_tensor.shape[1] + non_zeros_in_sparse

        if unit == "Byte":
            res = (self.num_element_weight + num_element)*element_size
            res_finetuned = num_element*element_size
        elif unit == "MB":
            res = ((self.num_element_weight + num_element)*element_size)/(1024*1024)
            res_finetuned = num_element*element_size/(1024*1024)
        elif unit == "KB":
            res = ((self.num_element_weight + num_element)*element_size)/(1024)
            res_finetuned = num_element*element_size/(1024)
        else:
            raise ValueError("Unit is not suitable")
        
        with open(os.path.join(self.logger.log_dir, f'weight_memory_{unit}.log'), "a") as file:
            file.write(str(self.current_epoch) + "\t" + str(float(res)) + "\t" + str(float(res_finetuned)) + "\n")

        #################### Calculate FLOPs
        num_flops_fw = 0
        num_flops_bw = 0
        num_flops_inference = 0
        layer_ranks = []

        input_shapes = th.tensor(self.size[3], device=device).float() # Shape: (#batches * num_of_finetune, các shape)
        input_shapes = input_shapes.view(num_batches, self.num_of_finetune, -1) # Shape: (#batches, num_of_finetune, các shape)
        input_shapes = input_shapes.permute(1, 0, 2) # Shape: (num_of_finetune, #batch, 4 shape)
        
        if self.backbone_name == "swinT":
            for layer_idx in range(weight_size_tensor.shape[0]):
                K = weight_size_tensor[layer_idx, :, 1]
                B, H, W, C = input_shapes[layer_idx, :, 0], input_shapes[layer_idx, :, 1], input_shapes[layer_idx, :, 2], input_shapes[layer_idx, :, 3] # Shape: Each element (#batch, 1)
                C_prime = weight_size_tensor[layer_idx, :, 0]

                WSI_fw_overhead = calculate_flops_SVD(C_prime, C) # Decompose weight using SVD
                WSI_fw_low_rank = B*H*2*W*K*(C+C_prime)

                num_flops_inference += WSI_fw_low_rank
                num_flops_fw += WSI_fw_overhead + WSI_fw_low_rank # Shape: (#batch, 1)
                num_flops_bw += B*H*2*W*K*(C_prime + C) +  C_prime*C*(2*B*H*W) # Shape: (#batch, 1)

                layer_ranks.append(int(th.mean(K).item()))
                
        elif self.backbone_name == "vit_b_32":
            for layer_idx in range(weight_size_tensor.shape[0]): # Iterate through each active layer
                K = weight_size_tensor[layer_idx, :, 1]

                B, N, I = input_shapes[layer_idx, :, 0], input_shapes[layer_idx, :, 1], input_shapes[layer_idx, :, 2] # Shape: Each element (#batch, 1)
                O = weight_size_tensor[layer_idx, :, 0]

                WSI_fw_overhead = calculate_flops_SVD(O, I) # Decompose weight using SVD
                WSI_fw_low_rank = B*2*N*K*(I+O)

                num_flops_inference += WSI_fw_low_rank
                num_flops_fw += WSI_fw_overhead + WSI_fw_low_rank # Shape: (#batch, 1)
                num_flops_bw += B*N*K*(2*I-1) + B*N*O*(2*K - 1) + O*I*(2*B*N-1)

                layer_ranks.append(int(th.mean(K).item()))

                
                save_dir = os.path.join(self.logger.log_dir, "eigenvalues")
                os.makedirs(save_dir, exist_ok=True)
                        
        
        num_flops_inference = (th.sum(num_flops_inference, dim=0)/num_flops_inference.shape[0]).item()
        num_flops_fw = (th.sum(num_flops_fw, dim=0)/num_flops_fw.shape[0]).item()
        num_flops_bw = (th.sum(num_flops_bw, dim=0)/num_flops_bw.shape[0]).item()

        with open(os.path.join(self.logger.log_dir, f'layer_ranks.txt'), "a") as file:
            file.write(str(self.current_epoch))
            for rank in layer_ranks:
                file.write("\t" + str(rank))
            file.write("\n")

        with open(os.path.join(self.logger.log_dir, f'inference_FLOPs.log'), "a") as file:
            file.write(str(self.current_epoch) + "\t" + str((num_flops_inference)) + "\n")

        with open(os.path.join(self.logger.log_dir, f'fw_FLOPs.log'), "a") as file:
            file.write(str(self.current_epoch) + "\t" + str((self.num_flops_fw + num_flops_fw)) + "\n")

        with open(os.path.join(self.logger.log_dir, f'bw_FLOPs.log'), "a") as file:
            file.write(str(self.current_epoch) + "\t" + str(num_flops_bw) + "\n")

        with open(os.path.join(self.logger.log_dir, f'total_FLOPs.log'), "a") as file:
            file.write(str(self.current_epoch) + "\t" + str((num_flops_fw + num_flops_bw)) + "\n")

    def configure_optimizers(self):
        if self.use_sgd:
            optimizer = th.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            if self.lr_warmup == 0:
                scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, self.anneling_steps, eta_min=0.1 * self.learning_rate)
            else:
                def _lr_fn(epoch):
                    if epoch < self.lr_warmup:
                        lr = self.init_lr_prod + (1 - self.init_lr_prod) / (self.lr_warmup - 1) * epoch
                    else:
                        e = epoch - self.lr_warmup
                        es = self.anneling_steps - self.lr_warmup
                        lr = 0.5 * (1 + np.cos(np.pi * e / es))
                    return lr
                scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_fn)
            sch = {
                "scheduler": scheduler,
                'interval': self.scheduler_interval,
                'frequency': 1
            }
            return [optimizer], [sch]
        optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                  lr=self.learning_rate, weight_decay=self.weight_decay, betas=(0.8, 0.9))
        return [optimizer]
    
    def update_optimizer(self):
        # Get the current optimizer
        optimizer = self.trainer.optimizers[0]

        for name in self.filter_cfgs["finetuned_layer"]:
            path_seq = name.split('.')
            target = reduce(getattr, path_seq, self) # Turn on gradient

            optimizer.add_param_group({
                'params': filter(lambda p: p.requires_grad, target.parameters())
            })

    
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, 
                   optimizer_closure, on_tpu=False, 
                   using_native_amp=False, using_lbfgs=False):
    
        # 1. Run forward + backward (compute grads)
        optimizer_closure()
        
        if not self.measure_perplexity_HOSVD_var:
            # 2. Update
            optimizer.step()
            # 3. Zero grad
            optimizer.zero_grad()

    def bn_eval(self):
        def f(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
            m.momentum = 1.0
        self.apply(f)

    def forward(self, x):
        if self.backbone_name == 'mcunet':
            feat = self.backbone(x)[-1]
            feat = self.pooling(feat)
            feat = feat.flatten(start_dim=1)
            logit = self.classifier(feat)
        else:
            logit = self.backbone(x)
        return logit
    
    def forward_llm(self, input_ids, attention_mask):
        logit = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return logit

    def training_step(self, train_batch, batch_idx):
        if self.set_bn_eval:
            self.bn_eval()
        
        if self.backbone_name == "TinyLlama":
            input_ids = train_batch['input_ids'].to(device=self.device)
            attention_mask = train_batch['attention_mask'].to(device=self.device)
            label = train_batch['label'].to(device=self.device)

            outputs = self.forward_llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            logits = outputs.logits

            pred_cls = th.argmax(logits, dim=1)
            acc = th.sum(pred_cls == label) / label.shape[0]
            loss = self.loss(logits.view(-1, self.num_classes), label.view(-1))

        else:
            img, label = train_batch['image'], train_batch['label']
            if img.shape[1] == 1:
                img = th.cat([img] * 3, dim=1)
            logits = self.forward(img)
            pred_cls = th.argmax(logits, dim=-1)
            acc = th.sum(pred_cls == label) / label.shape[0]
            loss = self.loss(logits, label)

        self.log("Train/Loss", loss)
        self.log("Train/Acc", acc)
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs): 
        with open(os.path.join(self.logger.log_dir, 'train_loss.log'), 'a') as f:
            mean_loss = th.stack([o['loss'] for o in outputs]).mean()
            f.write(f"{self.current_epoch} {mean_loss}")
            f.write("\n")

        with open(os.path.join(self.logger.log_dir, 'train_acc.log'), 'a') as f:
            mean_acc = th.stack([o['acc'] for o in outputs]).mean()
            f.write(f"{self.current_epoch} {mean_acc}")
            f.write("\n")

    def validation_step(self, val_batch, batch_idx):
        if self.backbone_name == "TinyLlama":
            input_ids = val_batch['input_ids'].to(device=self.device)
            attention_mask = val_batch['attention_mask'].to(device=self.device)
            label = val_batch['label'].to(device=self.device)

            outputs = self.forward_llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            logits = outputs.logits
            pred_cls = th.argmax(logits, dim=-1)
            acc = th.sum(pred_cls == label) / label.shape[0]

            return {'acc': acc, 'label': label}
        
        else:
            img, label = val_batch['image'], val_batch['label']
            if img.shape[1] == 1:
                img = th.cat([img] * 3, dim=1)
            logits = self.forward(img)
            probs = logits.softmax(dim=-1)
            pred = th.argmax(logits, dim=1)
            self.acc(probs, label)
            loss = self.loss(logits, label)
            self.log("Val/Loss", loss)
            return {'pred': pred, 'prob': probs, 'label': label}

    def validation_epoch_end(self, outputs):
        f = open(os.path.join(self.logger.log_dir, 'val.log'),
                 'a') if self.logger is not None else None
        acc = self.acc.compute()
        if self.logger is not None:
            f.write(f"{self.current_epoch} {acc}\n")
            f.close()
        self.log("Val/Acc", acc)
        self.log("val-acc", acc)
        self.acc.reset()