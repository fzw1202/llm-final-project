import torch
import random

class Sampler:
    def __init__(self, temperature: float, top_p: float, top_k: int):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if not (self.temperature >= 0 and self.temperature <= 1):
            self.temperature = 0
        if not (self.top_p >= 0 and self.top_p <= 1):
            self.top_p = 0.9

    def temperatureSampling(self, logits: torch.Tensor):
        if self.temperature == 0:
            return torch.argmax(logits, dim=-1)
        else:
            adjusted_logits = logits / self.temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)
            return adjusted_probs

    def topP(self, logits: torch.Tensor):
        batch_size = logits.size(0)
        results = []
        adjusted_probs = self.temperatureSampling(logits)
        for batch_idx in range(batch_size):
            sorted_probs, sorted_indices = torch.sort(adjusted_probs[batch_idx], dim=-1, descending=True)
            cumulative_prob = torch.cumsum(sorted_probs, dim=-1)
            cutoff_index = torch.where(cumulative_prob > self.top_p)[0][0].item() + 1
            filtered_probs = sorted_probs[:cutoff_index]
            filtered_probs /= filtered_probs.sum()
            filtered_probs = filtered_probs.cpu().detach().numpy().flatten()
            chosen_index = random.choices(range(len(filtered_probs)), weights=filtered_probs, k=1)[0]
            results.append(sorted_indices[chosen_index].item())
        return results

    def topK(self, logits: torch.Tensor):
        batch_size = logits.size(0)
        results = []
        adjusted_probs = self.temperatureSampling(logits)
        for batch_idx in range(batch_size):
            sorted_probs, sorted_indices = torch.sort(adjusted_probs[batch_idx], dim=-1, descending=True)
            filtered_probs = sorted_probs[:self.top_k]
            filtered_probs /= filtered_probs.sum()
            filtered_probs = filtered_probs.cpu().detach().numpy().flatten()
            chosen_index = random.choices(range(len(filtered_probs)), weights=filtered_probs.tolist(), k=1)[0]
            results.append(sorted_indices[chosen_index].item())
        return results
