from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch



def get_cifar10(batch_size, num_batches):
    # Define a simple transformation to match MobileNetV2's expected input
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load the CIFAR-10 dataset and select a single image
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    subset_indices = list(range(batch_size*num_batches))
    single_batch_dataset = Subset(dataset, subset_indices)  # Use only the first batch for simplicity
    dataloader = DataLoader(single_batch_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

#######################################################
import pickle

class Perplexity:
    def __init__(self, set_of_epsilons=[], perplexity=[], ranks=[], layer_mems=[], link='.'):
        self.set_of_epsilons = set_of_epsilons
        self.perplexity = perplexity
        self.link = link
        self.ranks = ranks
        self.layer_mems = layer_mems

    def save(self, link):
        with open(link, 'wb') as file:
            pickle.dump({
                'set_of_epsilons': self.set_of_epsilons,
                'perplexity': self.perplexity,
                'ranks': self.ranks,
                'layer_mems': self.layer_mems
            }, file)
        print(f'Perplexity is saved at {link}')

    def load(self, link):
        with open(link, 'rb') as file:
            data = pickle.load(file)
            self.set_of_epsilons = data['set_of_epsilons']
            self.perplexity = data['perplexity']
            self.ranks = data['ranks']
            self.layer_mems = data['layer_mems']
    
    def get_suitable_ranks(self, best_indices, num_of_finetuned):
        if num_of_finetuned == None or num_of_finetuned > len(self.layer_mems):
            print("[Perplexity class] Warning, num_of_finetuned is bigger than total number of layer or None, set it to be total number of layer")
            num_of_finetuned = len(self.layer_mems)

        suitable_ranks = []
        start_layer = len(self.layer_mems) - num_of_finetuned

        rank_idx = 0
        for layer_idx in range(start_layer, len(self.layer_mems)):
            suitable_ranks.append(self.ranks[layer_idx][best_indices[rank_idx]])
            rank_idx += 1
        
        return suitable_ranks
    
    def get_suitable_mems(self, best_indices, num_of_finetuned):
        if num_of_finetuned == None or num_of_finetuned > len(self.layer_mems):
            print("[Perplexity class] Warning, num_of_finetuned is bigger than total number of layer or None, set it to be total number of layer")
            num_of_finetuned = len(self.layer_mems)

        suitable_mems = []
        start_layer = len(self.layer_mems) - num_of_finetuned

        rank_idx = 0
        for layer_idx in range(start_layer, len(self.layer_mems)):
            suitable_mems.append(self.layer_mems[layer_idx][best_indices[rank_idx]])
            rank_idx += 1
        
        return suitable_mems

    def find_best_ranks_dp(self, budget, num_of_finetuned=None):
        round_to = 1000
        budget = int(budget * round_to)  

        if num_of_finetuned == None or num_of_finetuned > len(self.layer_mems):
            print("[Perplexity class] Warning, num_of_finetuned is bigger than total number of layer or None, set it to be total number of layer")
            num_of_finetuned = len(self.layer_mems)
        
        total_layer = len(self.layer_mems)
        start_layer = total_layer - num_of_finetuned
        num_ranks = len(self.layer_mems[0])

        min_budget_required = 0
        for layer in range(start_layer, start_layer + num_of_finetuned):
            min_layer_mem = min(self.layer_mems[layer])
            min_budget_required += min_layer_mem
        
        if min_budget_required * round_to > budget:
            print(f"[Warning] Budget is too small! Minimum required: {min_budget_required}, Given: {budget/round_to}")
            print("Set budget as minimum possible budget")
            budget = int(min_budget_required * round_to)
        
        dp = [[float('inf')] * (budget + 1) for _ in range(num_of_finetuned + 1)]
        dp[0][0] = 0
        
        choice = [[0] * (budget + 1) for _ in range(num_of_finetuned + 1)]
        
        for layer in range(1, num_of_finetuned + 1):
            for b in range(budget + 1):
                for rank in range(num_ranks):
                    mem_cost = int(self.layer_mems[start_layer + layer - 1][rank] * round_to)

                    if b >= mem_cost:
                        new_perplexity = dp[layer-1][b-mem_cost] + self.perplexity[start_layer + layer - 1][rank]

                        if new_perplexity < dp[layer][b]:
                            dp[layer][b] = new_perplexity
                            choice[layer][b] = rank
        
        best_perplexity = min(dp[num_of_finetuned])
        best_budget = dp[num_of_finetuned].index(best_perplexity)
        
        selected_ranks = []
        current_budget = best_budget
        
        for layer in range(num_of_finetuned, 0, -1):
            selected_rank = choice[layer][current_budget]
            selected_ranks.insert(0, selected_rank)
            current_budget -= int(self.layer_mems[start_layer + layer - 1][selected_rank] * round_to)
        
        best_budget_float = best_budget / float(round_to)
        
        return best_budget_float, best_perplexity, selected_ranks, self.get_suitable_ranks(selected_ranks, num_of_finetuned)