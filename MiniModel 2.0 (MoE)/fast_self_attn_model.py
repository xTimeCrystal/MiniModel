import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.cpp_extension import load
import gc
import os
import time
import math
import numpy as np
from functools import partial
from collections import Counter
from typing import Dict, List, Optional, Tuple, Callable, Union

from scattermoe.mlp import MLP as ScatterMoEMLP

eps = torch.finfo(torch.float32).eps

def norm(x: torch.Tensor):
    return torch.rms_norm(x, (x.size(-1),), eps=eps)

class Yarn(nn.Module):
    def __init__(self, config, max_seq_len, paired=False):
        super().__init__()
        self.head_dim = config.head_dim
        self.device = config.device
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=self.device)
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.device)
        if not self.paired:
            theta = torch.outer(t, angular_freq)
            self.factor1 = nn.Buffer(
                theta.cos().to(torch.bfloat16), persistent=False
            )
            self.factor2 = nn.Buffer(
                theta.sin().to(torch.bfloat16), persistent=False
            )
        else:
            t_even = 2 * t
            t_odd = 2 * t + 1
            theta1 = torch.outer(t_even, angular_freq)
            theta2 = torch.outer(t_odd, angular_freq)
            self.factor1 = nn.Buffer(
                torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
            self.factor2 = nn.Buffer(
                torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        if not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1.copy_(theta.cos())
            self.factor2.copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = 2 * t + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1.copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2.copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        
class CausalSoftmaxAttention(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        num_heads: int, 
        head_dim: int, 
    ):
        super().__init__()
        
        self.head_dim = head_dim
        self.num_heads = num_heads

        N = self.head_dim
        H = self.num_heads
        C = input_dim

        with torch.no_grad():
            init_bounds = 0.5 / (C ** 0.5)
    
            self.q_proj = nn.Linear(C, H*N, bias=False)
            self.k_proj = nn.Linear(C, H*N, bias=False)
            self.v_proj = nn.Linear(C, H*N, bias=False)
            self.g_proj = nn.Linear(C, H*N, bias=False)
            self.o_proj = nn.Linear(H*N, C, bias=False)
    
            self.q_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.k_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.v_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.g_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.o_proj.weight.data.zero_()

    def forward(self, x, yarn):
        B, T, C = x.size()
        N = self.head_dim
        H = self.num_heads

        def forward1(x):
            x = norm(x)
            
            q = self.q_proj(x).view(B, T, H, N)
            k = self.k_proj(x).view(B, T, H, N)
            v = self.v_proj(x).view(B, T, H, N)
            g = self.g_proj(x)

            q, k = yarn.rotary(norm(q)), yarn.rotary(norm(k))

            # k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]
            
            return (q, k, v, g)

        (q, k, v, g) = torch.utils.checkpoint.checkpoint(forward1, x, use_reentrant=False)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2), 
                k.transpose(1, 2), 
                v.transpose(1, 2), 
                is_causal=True, 
            ).transpose(1, 2).contiguous().view(B, T, H*N)

        x = self.o_proj(x * torch.sigmoid(g))
        
        return x
        
class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: Union[int, None] = None, 
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or 4 * input_dim

        with torch.no_grad():    
            init_bounds = 0.5 / (input_dim ** 0.5)
            
            self.k_proj = nn.Linear(input_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, input_dim)
            
            self.k_proj.weight.data.uniform_(-init_bounds, init_bounds)
            self.v_proj.weight.data.zero_()

    def forward(self, x):
        def forward1(x):
            x = norm(x)
            
            k = torch.relu(self.k_proj(x)).square()
            
            return self.v_proj(k).reshape(x.shape)

        output = torch.utils.checkpoint.checkpoint(forward1, x, use_reentrant=False)
        
        return output

class MoE(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        shared_expert: bool = True,
        renormalize: bool = True,
        router_aux_loss_coef: float = 1e-2,
    ):
        super().__init__()
        self.top_k = top_k
        self.shared_expert = shared_expert
        self.renormalize = renormalize
        self.router_aux_loss_coef = router_aux_loss_coef

        self.gate = nn.Linear(model_dim, num_experts, bias=False)

        self.experts = ScatterMoEMLP(
            input_size=model_dim,
            hidden_size=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            activation=lambda x: torch.relu(x).square()
        )
        
        if self.shared_expert:
            self.shared_gate = nn.Linear(model_dim, 1, bias=False)
            self.shared_expert = MLP(model_dim, ffn_dim)
        
        self.last_aux_loss: Optional[torch.Tensor] = None

    def _aux_loss(self, scores, topk_idx, aux_loss_coef):
        E = scores.size(-1)
        onehot = torch.zeros_like(scores).scatter_(1, topk_idx, 1.0 / self.top_k)
        return E * (onehot.mean(0) * scores.mean(0)).sum() * aux_loss_coef

    def forward(self, x):
        B, T, C = x.shape

        def forward1(x):
            x = norm(x.view(-1, C))
            raw_scores = self.gate(x)
            scores = torch.softmax(raw_scores, dim=-1)
            topk_w, topk_idx = scores.topk(self.top_k, dim=-1)
            
            if self.renormalize:
                topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-6)
    
            aux_loss = self._aux_loss(scores, topk_idx, self.router_aux_loss_coef)
    
            # ScatterMoE signature: mlp(X, k_weights, k_idxs)
            out = self.experts(x, topk_w, topk_idx)

            if self.shared_expert:
                out = out + self.shared_gate(x).sigmoid() * self.shared_expert(x)
                
            return out, aux_loss
            
        out, aux_loss = torch.utils.checkpoint.checkpoint(forward1, x, use_reentrant=False)

        self.last_aux_loss = aux_loss
        
        return out.view(B, T, C)

class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = CausalSoftmaxAttention(config.model_dim, config.num_heads, config.head_dim)
        self.moe = MoE(config.model_dim, config.hidden_dim, 
                       config.num_experts, config.top_k,
                       shared_expert=True, renormalize = True, 
                       router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 1e-3))
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x, yarn):
        xx = self.att(x, yarn)
        x = x + xx
        
        xx = self.moe(x)
        x = x + xx
        
        self.last_aux_loss = self.moe.last_aux_loss
        return x
        
class SoftmaxBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.att = CausalSoftmaxAttention(config.model_dim, config.num_heads, config.head_dim)
        self.ffn = MLP(config.model_dim, config.hidden_dim)
    
    def forward(self, x, yarn):
        xx = self.att(x, yarn)
        x = x + xx
        
        xx = self.ffn(x)
        x = x + xx
        
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.emb = nn.Embedding(config.vocab_size, config.model_dim).to(config.device)
        self.emb.weight.data.uniform_(-1e-4, 1e-4)
        
        self.yarn = Yarn(config, 2048)
        
        # self.blocks = nn.ModuleList([SoftmaxBlock(config) for i in range(config.layers)]).to(config.device)
        self.blocks = nn.ModuleList([MoEBlock(config) for i in range(config.layers)]).to(config.device)

    def forward(self, idx):
        x = norm(self.emb(idx))

        aux_loss = torch.tensor(0.0, device=self.config.device)
        
        for i, block in enumerate(self.blocks):
            x = block(x, self.yarn)
            aux_loss += block.last_aux_loss
            
        x = norm(x)
            
        if self.training:
            return x, aux_loss
        else:
            logits = F.linear(x, self.emb.weight)
            return logits
