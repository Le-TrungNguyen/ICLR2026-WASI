import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights, vit_b_32, ViT_B_32_Weights
# import timm
from custom_op.register import register_normal_linear, register_lora, register_ASI, register_WASI
# from custom_op_cpp.register import register_WASI
from utils import Perplexity
from custom_op.linear.linear_lora import LoRALinear

class ModelTrainer:
    def __init__(self, model_name, batch_size, num_epochs, device='cuda',
                 rank=None, 
                 with_base=False, with_WASI=False, with_lora=False, with_ASI=False, budget=None,
                 dataloader=None, output_channels=None, num_of_finetune=None, 
                 perplexity_link=None, explained_var=None, checkpoint=None, energy_logger=None, micro_benchmark_wasi=False):
        
        self.device = device
        self.model_name = model_name
        self.output_channels = output_channels
        self.model_dict = self.get_model(model_name, checkpoint)
        
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.num_epochs = num_epochs

        # if self.model_name == "swinT" or self.model_name == "vit_b_32" or self.model_name == "vit_tiny_patch16_224":
        self.all_linear_layers = self.get_all_linear_with_name()
        if num_of_finetune == "all" or num_of_finetune > len(self.all_linear_layers):
            print("[Warning] Finetuning all layers")
            self.num_of_finetune = len(self.all_linear_layers)
        else:
            self.num_of_finetune = num_of_finetune


        self.budget = budget
        self.rank = rank
        self.explained_var = explained_var
        self.with_base = with_base
        self.with_WASI = with_WASI
        self.with_lora = with_lora
        self.with_ASI = with_ASI

        self.perplexity_link = perplexity_link

        self.backward_time = []
        self.forward_time = []
        self.inference_time = []

        ########## Micro-benchmark time of WASI ##########
        if micro_benchmark_wasi:
            self.output_calculation_time = []

            ## Time of performing WSI
            self.orthogonalization_time = []
            self.matmuls_time = []
        else:
            self.output_calculation_time = None

            ## Time of performing WSI
            self.orthogonalization_time = None
            self.matmuls_time = None



        self.energy_logger = energy_logger

        # self.model_dict = self.get_model(model_name)
        self.config_model(self.backward_time, self.forward_time, self.inference_time, self.energy_logger,
                          self.output_calculation_time, self.orthogonalization_time, self.matmuls_time)


    def get_all_linear_with_name(self):
        linear_layers = {}
        for name, mod in self.model_dict['model'].named_modules():
            if isinstance(mod, nn.modules.linear.Linear) and 'mlp' in name:
                linear_layers[name] = mod
        return linear_layers
    
    def get_all_linear_with_name(self):
        linear_layers = {}
        visited = set()

        for name, mod in self.model_dict['model'].named_modules():
            if any(name.startswith(v + ".") for v in visited):
                continue

            if isinstance(mod, LoRALinear) and 'mlp' in name:
                linear_layers[name] = mod
                visited.add(name)
            elif isinstance(mod, nn.modules.linear.Linear) and 'mlp' in name:
                linear_layers[name] = mod

        return linear_layers
    
    def get_model(self, model_name, checkpoint):
        if model_name == 'swinT': model = swin_t(weights=Swin_T_Weights.DEFAULT)
        elif model_name == 'vit_b_32': 
            if checkpoint is not None:
                pruned_dict = torch.load(checkpoint, weights_only=False, map_location=self.device)
                model = pruned_dict['model']
            else: model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)

        # Modify the last layer for CIFAR-10 (10 classes)
        if model_name == 'swinT':
            model.head = nn.Linear(in_features=768, out_features=self.output_channels, bias=True) # Change classifier
        elif model_name == 'vit_b_32':
            model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=self.output_channels, bias=True)) # Change classifier

        model.to(self.device)
        
        return {"model": model, "name": model_name}

    


    def freeze_layers(self, num_of_finetune):
        # if self.model_name != 'swinT' and self.model_name != 'vit_b_32':
        #     return
        
        all_layers = self.all_linear_layers

        finetuned_layers = dict(list(all_layers.items())[-num_of_finetune:])
        for name, mod in self.model_dict['model'].named_modules():
            if len(list(mod.children())) == 0 and name not in finetuned_layers.keys() and name != '':
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False # Freeze layer
            elif name in finetuned_layers.keys():
                break
        return finetuned_layers

    def config_model(self, backward_time, forward_time, inference_time, energy_logger,
                     output_calculation_time, orthogonalization_time, matmuls_time):
        
        finetuned_layers = self.freeze_layers(self.num_of_finetune)
        filter_cfgs = { "finetuned_layer": finetuned_layers, 
                        "type": "conv",
                        "backward_time": backward_time,
                        "forward_time": forward_time,
                        "inference_time": inference_time,
                        "energy_logger": energy_logger,
                        "output_calculation_time": output_calculation_time,
                        "orthogonalization_time": orthogonalization_time,
                        "matmuls_time": matmuls_time}
        if self.with_base: 
            if self.model_name == 'swinT' or self.model_name == 'vit_b_32':
                filter_cfgs["type"] = "linear"
                register_normal_linear(self.model_dict['model'], filter_cfgs)
        elif self.with_WASI:
            if self.perplexity_link is None:
                self.suitable_ranks = [1] * len(self.all_linear_layers)
            else:
                perplexity = Perplexity()
                perplexity.load(self.perplexity_link)
                best_memory, best_perplexity, best_indices, self.suitable_ranks = perplexity.find_best_ranks_dp(budget=0.01, num_of_finetuned=self.num_of_finetune)
                del perplexity
            filter_cfgs["activation_ranks"] = self.suitable_ranks
            filter_cfgs["explained_variance_threshold"] = self.explained_var
            filter_cfgs["type"] = "linear"

            register_WASI(self.model_dict['model'], filter_cfgs)
        elif self.with_lora:
            filter_cfgs["rank"] = self.rank
            register_lora(self.model_dict['model'], filter_cfgs)
        elif self.with_ASI:
            perplexity = Perplexity()
            perplexity.load(self.perplexity_link)
            best_memory, best_perplexity, best_indices, self.suitable_ranks = perplexity.find_best_ranks_dp(budget=self.budget, num_of_finetuned=self.num_of_finetune)
            del perplexity
            
            filter_cfgs["activation_ranks"] = self.suitable_ranks
            filter_cfgs["type"] = "linear"

            register_ASI(self.model_dict['model'], filter_cfgs)



    def train_model(self):
        print("Training begin ...", end='')
        optimizer = torch.optim.SGD(self.model_dict['model'].parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        if self.energy_logger is not None:
            self.energy_logger.start_phase("train_energy")
            self.energy_logger.pause()

        for epoch in range(self.num_epochs):
            self.model_dict['model'].train()
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model_dict['model'](inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("Done", end='\n')

        if self.energy_logger is not None:
            self.energy_logger.stop_phase()
    
    def inference_model(self, dataloader=None):
        print("Inference begin ...", end='')

        if dataloader is None:
            dataloader = self.dataloader

        self.model_dict['model'].eval()
        predictions = []
        ground_truths = []

        if self.energy_logger is not None:
            self.energy_logger.start_phase("inference_energy")

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model_dict['model'](inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                ground_truths.extend(labels.cpu().numpy())
        print("Done", end='\n')

        if self.energy_logger is not None:
            self.energy_logger.stop_phase()

        return predictions, ground_truths


    def warmup_model(self, warmup_steps=5):
        optimizer = torch.optim.SGD(self.model_dict['model'].parameters(), lr=0.001)
        
        criterion = nn.CrossEntropyLoss()

        print(f"Starting warm-up for {warmup_steps} steps...")
        self.model_dict['model'].train()
        for i, (inputs, labels) in enumerate(self.dataloader):
            if i >= warmup_steps:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # Forward pass
            outputs = self.model_dict['model'](inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Warm-up completed.")
