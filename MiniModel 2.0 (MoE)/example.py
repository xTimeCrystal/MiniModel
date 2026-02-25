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
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from functools import partial
from collections import Counter
from typing import Dict, List, Optional, Tuple, Callable, Union

import torch._inductor.config as inductor_config

inductor_config.fx_graph_cache = True

os.environ["TRITON_CACHE_DIR"] = "~/.triton/cache"

from .data_utils import load_parquet
from .fast_optim  import AdaMuon
from .fast_self_attn_model import Transformer as Model
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from torch.profiler import profile, record_function, ProfilerActivity
import contextlib

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 64

class GPTConfig:
    layers = 12
    vocab_size = 32768

    model_dim = 512
    hidden_dim = 512
    num_experts = 128
    top_k = 4
    router_aux_loss_coef = 1e-3
    
    num_heads = 4
    head_dim = 128
    
    dtype = torch.bfloat16

    device = 'cuda:0'

config = GPTConfig()

batch_size = 128
seq_length = 2048
grad_steps = 1

base_lr      = 1e-3
anneal_lr    = base_lr/10

anneal_steps = 1_200 * 8

model = Model(config)
model.zero_grad()
model.bfloat16()

# model = convert_to_float8_training(model)

dtype = GPTConfig.dtype

# saved_states = torch.load('rwkv_pt/TinyCorpus_1535387_0.8046875_Jul_24_2025_checkpoint.pt', weights_only=False)
# model.load_state_dict(saved_states['model'])

param_shapes = set([p.shape for p in model.parameters()])

optimizer = AdaMuon([{'params': [p for p in model.parameters() if (p.shape == shape)]} for shape in param_shapes], 
                    lr=base_lr, betas=(0.65, 0.95), weight_decay=0.0)

# optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.65, 0.95), weight_decay=0.0, fused=True)

scheduler = optim.swa_utils.SWALR(optimizer, anneal_lr, anneal_steps)

@torch.compile
def forward_pass(model, X):
    return model(X)

@torch._dynamo.disable
def loss_fn(model, X):
    Y = torch.roll(X, -1, dims=1)
    Y[:, -1] = -100
    
    logits, lb_loss = forward_pass(model, X)
    
    loss_func = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean")
    loss = loss_func(
        model.emb.weight, 
        logits.view(-1, logits.shape[-1]), 
        Y.view(-1)
    )
    return loss, lb_loss

# Training Loop

cur_step = 0

def checkpoint(cur_step, loss, save_states):
    model, optimizer = save_states
    
    dataset_name = 'TinyCorpus'
    date = time.strftime("%b_%d_%Y", time.gmtime())
    file_name = f'{dataset_name}_{cur_step}_{loss}_{date}'

    checkpoint = {
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
    }
    
    torch.save(checkpoint, f'{file_name}_checkpoint.pt')
    torch.save(model.state_dict(), f'{file_name}_weights.pt')

for dataset_n in range(1):

    torch.cuda.empty_cache()
    gc.collect()
    
    # Load Data
    
    base_path = "datasets/TinyCorpus/128/"
    data_name = f"tinycorpus-{dataset_n%480:03d}-of-128.parquet"

    loader, n_rows = load_parquet(base_path, data_name, batch_size=batch_size, columns=['0'])

    print(f'Loaded {data_name}')
    
    pbar = tqdm(total=n_rows*seq_length, unit="tokens")

    torch.cuda.synchronize()
    
    # Init
    model.train()

    torch.cuda.empty_cache()
    
    for mini_batch, data in enumerate(loader):
        cpu_tensor = torch.tensor(data['0'], pin_memory=True)
        tok_batch = cpu_tensor.to(device=config.device, non_blocking=True)

        loss, lb_loss = loss_fn(model, tok_batch)

        if torch.isnan(loss).item():
            states = (model, optimizer)
            checkpoint(cur_step, 'NaN', states)
            assert False, 'NaNs encountered during training'
             
        (loss+lb_loss).backward()
        loss = loss.item()
        lb_loss = lb_loss.item()

        optimizer.step()
        scheduler.step()

        model.zero_grad(set_to_none=True)
        
        writer.add_scalar("Loss/train", loss, cur_step)
        writer.add_scalar("Loss/train_moe", lb_loss, cur_step)
        
        torch.cuda.reset_peak_memory_stats(device='cuda')

        pbar.update(batch_size*seq_length)
        
        cur_step += 1

    loader.cleanup()
    gc.collect()
    del loader

    writer.flush()
    states = (model, optimizer)
    checkpoint(cur_step, loss, states)
    
    model.eval()
