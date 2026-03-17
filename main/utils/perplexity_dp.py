import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

class Perplexity:
    def __init__(self, set_of_epsilons=[], perplexity=[], ranks=[], layer_mems=[], link='.'):
        self.set_of_epsilons = set_of_epsilons
        self.perplexity = perplexity
        self.link = link 
        self.ranks = ranks 
        self.layer_mems = layer_mems 
        

    def plot(self, is_saved=False, name=None):
        if not self.set_of_epsilons or not self.perplexity:
            print("Dữ liệu epsilon hoặc perplexity trống.")
            return

        for layer_index, layer_perplexity in enumerate(self.perplexity):
            plt.plot(self.set_of_epsilons, layer_perplexity, label=f'Layer {layer_index + 1}')

        plt.xlabel('Epsilon')
        plt.ylabel('Perplexity')
        plt.title('Perplexity vs Epsilon for Different Layers')
        plt.legend()
        plt.grid(True)

        plt.xticks(self.set_of_epsilons)

        if is_saved:
            if name is None: name = 'perplexity.svg'
            file_path = os.path.join(self.link, name)
            plt.savefig(file_path)
            print(f'Figure is saved at {file_path}')

        plt.show()

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

def merged_perplexity(*links_to_perplexity):
    if not links_to_perplexity:
        raise ValueError("No perplexity files provided!")

    perplexity_merged = Perplexity()
    perplexity_merged.load(links_to_perplexity[0])

    for link in links_to_perplexity[1:]:
        perplexity_temp = Perplexity()
        perplexity_temp.load(link)

        perplexity_merged.set_of_epsilons += perplexity_temp.set_of_epsilons
        perplexity_merged.perplexity = [row1 + row2 for row1, row2 in zip(perplexity_merged.perplexity, perplexity_temp.perplexity)]
        perplexity_merged.ranks = [row1 + row2 for row1, row2 in zip(perplexity_merged.ranks, perplexity_temp.ranks)]
        perplexity_merged.layer_mems = [row1 + row2 for row1, row2 in zip(perplexity_merged.layer_mems, perplexity_temp.layer_mems)]

    saved_location = os.path.dirname(os.path.dirname(links_to_perplexity[0]))
    os.makedirs(saved_location, exist_ok=True)
    saved_file = os.path.join(saved_location, 'perplexity_combined.pkl')
    
    perplexity_merged.save(saved_file)
    print(f"Perplexity is saved at {saved_file}")

    return saved_file

