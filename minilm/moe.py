"""Sparse Mixture of Experts (MoE) layer.

Covers:
- Session 13: Beyond dense Transformers (MoE, top-k routing, load balancing loss).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import GPTConfig, MLP


class TopKRouter(nn.Module):
    """Router projecting token embeddings to expert logits and selecting top-k experts."""

    def __init__(self, n_embd: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(n_embd, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, n_embd] -> flat: [B*T, n_embd]
        B, T, C = x.size()
        flat_x = x.view(-1, C)
        logits = self.gate(flat_x)  # [B*T, num_experts]

        # Top-k selection
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)  # [B*T, top_k]

        # Auxiliary load-balancing loss calculation (Switch Transformer / GShard style)
        # P_i = mean router probability assigned to expert i
        # f_i = fraction of tokens routed to expert i
        probs = F.softmax(logits, dim=-1)  # [B*T, num_experts]
        P = probs.mean(dim=0)

        # Count tokens dispatched per expert
        mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1)  # [B*T, num_experts]
        f = mask.float().mean(dim=0)

        # Auxiliary loss = num_experts * sum(f_i * P_i)
        aux_loss = self.num_experts * torch.sum(f * P)

        return topk_weights, topk_indices, aux_loss


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts (MoE) replacing a dense MLP."""

    def __init__(self, config: GPTConfig, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(config.n_embd, num_experts, top_k=top_k)
        self.experts = nn.ModuleList([MLP(config) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        flat_x = x.view(-1, C)

        weights, indices, aux_loss = self.router(x)
        final_output = torch.zeros_like(flat_x)

        # Dispatch tokens to selected experts
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens assigned to this expert across top_k choices
            # mask: [B*T, top_k]
            selected = (indices == expert_idx)
            if not selected.any():
                continue

            # Token indices: token_rows has indices in 0 .. B*T-1
            token_rows, k_pos = torch.where(selected)
            expert_in = flat_x[token_rows]
            expert_out = expert(expert_in)

            # Weight by router probability
            token_weights = weights[token_rows, k_pos].unsqueeze(-1)
            final_output.index_add_(0, token_rows, expert_out * token_weights)

        return final_output.view(B, T, C), aux_loss
