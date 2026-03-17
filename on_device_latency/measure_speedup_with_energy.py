import torch
import argparse
from model_trainer import ModelTrainer
from utils import get_cifar10
import gc
from energy_logger import EnergyLogger

parser = argparse.ArgumentParser(description='Run model training and evaluation.')
parser.add_argument('explained_var', type=float, help='Explained variance threshold')
parser.add_argument('--budget_ASI', type=float, help='Budget for ASI')
parser.add_argument('--method', type=str, choices=['vanilla', 'WASI', 'ASI'], default='vanilla', help='Method to use: vanilla or WASI or ASI')

parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                    choices=['cpu', 'cuda'], help='Device to use for training (default: cuda if available)')

parser.add_argument('--model_name', type=str, default='vit_b_32', help='vit_b_32 or swinT')

args = parser.parse_args()
num_of_finetune = "all"
explained_var = args.explained_var
budget_ASI = args.budget_ASI
method = args.method
device = torch.device(args.device)
model_name = args.model_name

print("---------------------------------")
print("explained_var = ", explained_var, " | method = ", args.method)

# Configurations:
torch.manual_seed(233)
batch_size = 128
num_batches = 1
num_epochs = 1

# Get data
dataloader = get_cifar10(batch_size, num_batches)

# ================== Energy logger setup (Jetson Orin) ==================
energy_logger = None
try:
    # đo trên rail CPU+GPU; nếu muốn thêm VDD_IN thì rails_to_use=["VDD_CPU_GPU_CV", "VDD_IN"]
    energy_logger = EnergyLogger(interval=0.05, rails_to_use=["VDD_CPU_GPU_CV"])
    energy_logger.start_global()
    print("[EnergyLogger] Started.")
except Exception as e:
    print("[EnergyLogger] Disabled:", e)
    energy_logger = None
# ======================================================================

if method == 'vanilla':
    ######################## Vanilla ############################
    # Get model
    model = ModelTrainer(model_name, batch_size, num_epochs, device=device,
                    with_base=True, dataloader=dataloader, output_channels=10, num_of_finetune=num_of_finetune, energy_logger=energy_logger)

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
                    with_WASI=True, dataloader=dataloader, output_channels=10, explained_var=explained_var, num_of_finetune=num_of_finetune, perplexity_link=perplexity_link[model_name], energy_logger=energy_logger)
    model.train_model()
    model.inference_model()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

elif method == 'ASI':
    ######################### ASI ##########################
    # Get model
    perplexity_link = {'vit_b_32':"perplexity/vit_b_32_perplexity.pkl",
                       'swinT':'perplexity/swinT_perplexity.pkl'}
    model = ModelTrainer(model_name, batch_size, num_epochs, device=device,
                    with_ASI=True, dataloader=dataloader, output_channels=10, budget=budget_ASI, num_of_finetune=num_of_finetune, perplexity_link=perplexity_link[model_name], energy_logger=energy_logger)
    model.train_model()
    model.inference_model()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ================== Stop energy logger & collect ==================
energies = None
if energy_logger is not None:
    energy_logger.stop_global()
    energies = energy_logger.get_energy()   # dict {phase: {rail: energy_J}}
    print("[EnergyLogger] Stopped.")
# =================================================================

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

check_and_write_title(f'{processed_time_folder}/time.txt', "explained_var\tinference_time\ttraining_time\ttraining_time_forward\ttraining_time_backward\tmethod\n")
# Log results
with open(f'{processed_time_folder}/time.txt', "a") as file:
    file.write(f"{explained_var}\t{sum(model.inference_time)}\t{sum(model.forward_time) + sum(model.backward_time)}\t{sum(model.forward_time)}\t{sum(model.backward_time)}\t{method}\n")

# ============= Energy logs =============
if energies is not None:
    def get_phase_energy(phase, rail="VDD_CPU_GPU_CV"):
        if phase not in energies:
            return 0.0
        return energies[phase].get(rail, 0.0)

    times = energy_logger.get_time()  # {phase: time_seconds}

    def get_phase_time(phase):
        return times.get(phase, 0.0)

    # Energy (J)
    train_energy = get_phase_energy("train_energy")
    inference_energy = get_phase_energy("inference_energy")

    # Time (s)
    train_time = get_phase_time("train_energy")
    inference_time = get_phase_time("inference_energy")

    # Power (W) = J / s
    train_power = train_energy / train_time if train_time > 0 else 0.0
    inference_power = inference_energy / inference_time if inference_time > 0 else 0.0

    print("Train:   E =", train_energy, "J  |  t =", train_time, "s  |  P =", train_power, "W")
    print("Infer:   E =", inference_energy, "J  |  t =", inference_time, "s  |  P =", inference_power, "W")

    # Log energy
    check_and_write_title(f'{processed_time_folder}/energy.txt', "explained_var\tinference_energy_J\ttraining_energy_J\tmethod\n")

    with open(f'{processed_time_folder}/energy.txt', "a") as f:
        f.write(f"{explained_var}\t{inference_energy}\t{train_energy}\t{method}\n")

    # Log power (W)
    check_and_write_title(f'{processed_time_folder}/power.txt', "explained_var\tinference_power_W\ttraining_power_W\tmethod\n")

    with open(f'{processed_time_folder}/power.txt', "a") as f:
        f.write(f"{explained_var}\t{inference_power}\t{train_power}\t{method}\n")
# ================================================