import torch
import argparse
from model_trainer import ModelTrainer
from utils import get_cifar10
import gc

parser = argparse.ArgumentParser(description='Run model training and evaluation.')
parser.add_argument('explained_var', type=float, help='Explained variance threshold')
parser.add_argument('--budget_ASI', type=float, help='Budget for ASI')
parser.add_argument('--method', type=str, choices=['vanilla', 'WASI', 'ASI'], default='vanilla', help='Method to use: vanilla or WASI or ASI')

parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                    choices=['cpu', 'cuda'], help='Device to use for training (default: cuda if available)')

parser.add_argument('--model_name', type=str, default='vit_b_32', help='vit_b_32 or swinT')
parser.add_argument('--num_of_finetune', type=str, default="all", help='number of layer being finetuned')

args = parser.parse_args()
num_of_finetune = args.num_of_finetune if args.num_of_finetune == 'all' else int(args.num_of_finetune)
explained_var = args.explained_var
budget_ASI = args.budget_ASI
method = args.method
device = torch.device(args.device)
model_name = args.model_name

micro_benchmark_wasi = True

print("---------------------------------")
print("explained_var = ", explained_var, " | method = ", args.method)

# Configurations:
torch.manual_seed(233)
batch_size = 128
num_batches = 1
num_epochs = 1

# Get data
dataloader = get_cifar10(batch_size, num_batches)

if method == 'vanilla':
    ######################## Vanilla ############################
    # Get model
    model = ModelTrainer(model_name, batch_size, num_epochs, device=device,
                    with_base=True, dataloader=dataloader, output_channels=10, num_of_finetune=num_of_finetune)

    model.train_model()
    model.inference_model()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

elif method == 'WASI':
    ######################### WASI ##########################
    # Get model
    perplexity_link = {'vit_b_32':"perplexity/vit_b_32_perplexity.pkl",
                       'swinT':'perplexity/swinT_perplexity.pkl'}
    model = ModelTrainer(model_name, batch_size, num_epochs, device=device,
                    with_WASI=True, dataloader=dataloader, output_channels=10, explained_var=explained_var, num_of_finetune=num_of_finetune, perplexity_link=perplexity_link.get(model_name, None), micro_benchmark_wasi=micro_benchmark_wasi)
    model.train_model()
    model.inference_model()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# elif method == 'ASI':
#     ######################### ASI ##########################
#     # Get model
#     perplexity_link = {'vit_b_32':"perplexity/vit_b_32_perplexity.pkl",
#                        'swinT':'perplexity/swinT_perplexity.pkl'}
#     model = ModelTrainer(model_name, batch_size, num_epochs, device=device,
#                     with_ASI=True, dataloader=dataloader, output_channels=10, budget=budget_ASI, num_of_finetune=num_of_finetune, perplexity_link=perplexity_link.get(model_name, None))
#     model.train_model()
#     model.inference_model()

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#     gc.collect()

import os
os.makedirs(f'processed_time_{args.device}', exist_ok=True)

# Create result folder
processed_time_folder = f"processed_time_{args.device}/batch_size={batch_size}/{model_name}"
if not os.path.exists(processed_time_folder):
    os.makedirs(processed_time_folder)

def check_and_write_title(file_path, title):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "a") as file:
            file.write(title)

if not micro_benchmark_wasi:
    check_and_write_title(f'{processed_time_folder}/time.txt', "explained_var\tinference_time\ttraining_time\ttraining_time_forward\ttraining_time_backward\tmethod\n")
    # Log results
    with open(f'{processed_time_folder}/time.txt', "a") as file:
        file.write(f"{explained_var}\t{sum(model.inference_time)}\t{sum(model.forward_time) + sum(model.backward_time)}\t{sum(model.forward_time)}\t{sum(model.backward_time)}\t{method}\n")

elif micro_benchmark_wasi:
    check_and_write_title(f'{processed_time_folder}/time.txt', "explained_var\tinference_time\ttraining_time\ttraining_time_forward\toutput_calculation_time\tmatmuls_time\torthogonalization_time\ttraining_time_backward\tmethod\n")
    # Log results
    if method != 'WASI':
        with open(f'{processed_time_folder}/time.txt', "a") as file:
            file.write(f"{explained_var}\t{sum(model.inference_time)}\t{sum(model.forward_time) + sum(model.backward_time)}\t{sum(model.forward_time)}\tNone\tNone\tNone\t{sum(model.backward_time)}\t{method}\n")
    else:
        with open(f'{processed_time_folder}/time.txt', "a") as file:
            file.write(f"{explained_var}\t{sum(model.inference_time)}\t{sum(model.forward_time) + sum(model.backward_time)}\t{sum(model.forward_time)}\t{sum(model.output_calculation_time)}\t{sum(model.matmuls_time)}\t{sum(model.orthogonalization_time)}\t{sum(model.backward_time)}\t{method}\n")