import gc
import os
import time
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from functools import partial
from collections import Counter
from typing import Dict, List, Optional, Tuple, Callable, Union

class LMDatasetLoader:
    def __init__(self, mb_sz, mb_len, dataset, density=0.75):
        self.mb_sz, self.mb_len, self.dataset = mb_sz, mb_len, dataset
        self.current_texts = [[] for _ in range(self.mb_sz)]
        self.density = density
        self.n = 0
        self.max_n = len(self.dataset)
        
        for i in range(self.mb_sz):
            if self.n >= self.max_n:
                raise ValueError('Length of dataset must be greater than minibatch size')
            self.current_texts[i] += list(self.dataset[self.n])
            self.n += 1
            
    def __iter__(self):
        return self

    def __next__(self):
        gates = []

        max_length = self.mb_len
        
        for i in range(self.mb_sz):
            
            if self.n >= self.max_n:
                self.cleanup()
                raise StopIteration
                
            if len(self.current_texts[i]) == 0:
                gates.append(0)
                self.current_texts[i] += list(self.dataset[self.n])
                self.n += 1
            else:
                gates.append(1)
                
            if self.n >= self.max_n:
                self.cleanup()
                raise StopIteration
                
            while (len(self.current_texts[i]) + len(self.dataset[self.n]) <= self.mb_len or 
                   len(self.current_texts[i]) < self.mb_len * self.density):
                
                self.current_texts[i] += list(self.dataset[self.n])
                self.n += 1
                
                if self.n >= self.max_n:
                    self.cleanup()
                    raise StopIteration

        outputs = [text[:self.mb_len] + [0]*(max_length-len(text[:self.mb_len])) for text in self.current_texts]
        true_lengths = [len(text[:self.mb_len]) for text in self.current_texts]
        
        self.current_texts = [text[self.mb_len:] for text in self.current_texts]
        return (outputs, gates, true_lengths)

    def __len__(self):
        remaining_dataset = sum(len(self.dataset[i]) for i in range(self.n, self.max_n))
        current_data = sum(len(text) for text in self.current_texts)
        total_data = remaining_dataset + current_data
        return math.ceil(total_data / (self.density*self.mb_len)) * self.mb_len

    def cleanup(self):
        if hasattr(self, 'current_texts') and self.current_texts is not None:
            for text_list in self.current_texts:
                text_list.clear()
            self.current_texts.clear()
            self.current_texts = None
        
        if hasattr(self, 'dataset') and self.dataset is not None:
            if hasattr(self.dataset, 'clear'):
                self.dataset.clear()
            self.dataset = None
        
        gc.collect()

    def __del__(self):
        self.cleanup()

class DatasetLoader:
    def __init__(self, batch_size: int, dataset: list, columns: list = None, maps: dict = {}):
        self.batch_size, self.dataset = batch_size, dataset
        
        self.columns = columns or set([x for r in dataset for x in r.keys()])
        self.maps = dict([(key, maps[key]) if key in maps.keys() else (key, lambda x: x) for key in self.columns])
        
        self.n = 0
        self.max_n = len(self.dataset)
            
    def __iter__(self):
        return self

    def __next__(self):
        outputs = dict([(key, []) for key in self.columns])
        
        for i in range(self.batch_size):
            
            if self.n >= self.max_n:
                self.cleanup()
                raise StopIteration

            for key in self.columns:
                outputs[key].append(self.maps[key](self.dataset[self.n][key]))
                
            self.n += 1
                
        return outputs

    def cleanup(self):        
        if hasattr(self, 'dataset') and self.dataset is not None:
            if hasattr(self.dataset, 'clear'):
                self.dataset.clear()
            self.dataset = None
        
        gc.collect()

    def __del__(self):
        self.cleanup()

def lm_load_parquet(base_path, dataset_name, batch_size, seq_length, column='0'):
    full_path = f"{base_path}{dataset_name}"
    table = pq.read_table(full_path, columns=[column])
    
    buffer = table[column].to_pylist()
    
    del table
    
    try:
        pa.default_memory_pool().release_unused()
    except:
        pass
    
    gc.collect()
    
    n_bytes = sum([len(item) for item in buffer])

    return LMDatasetLoader(batch_size, seq_length, buffer), n_bytes

def load_parquet(base_path, dataset_name, batch_size: int, columns: list, maps: dict = {}):
    full_path = f"{base_path}{dataset_name}"
    table = pq.read_table(full_path, columns=columns)
    
    buffer = table.to_pylist()
    
    del table
    
    try:
        pa.default_memory_pool().release_unused()
    except:
        pass
    
    gc.collect()

    return DatasetLoader(batch_size, buffer, columns, maps), len(buffer)

import gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sortedcontainers import SortedList

# ==========================================
# 1. Exact Packing Algorithm Provided
# ==========================================
def pack_best_fit_sorted(tokenized_dataset, capacity=2048):
    bins = []
    # SortedList of tuples (remaining_space, bin_index)
    free = SortedList()

    for doc in tokenized_dataset:
        doc_len = len(doc)
        # find first bin with remaining_space >= doc_len
        idx = free.bisect_left((doc_len, -1))  # -1 to find the earliest such tuple
        if idx < len(free):
            rem, bin_idx = free.pop(idx)
            bins[bin_idx].extend(doc)
            new_rem = rem - doc_len
            free.add((new_rem, bin_idx))
        else:
            # new bin
            bin_idx = len(bins)
            bins.append(list(doc))
            free.add((capacity - doc_len, bin_idx))

    return bins

# ==========================================
# 2. PyTorch-Safe Packed Loader
# ==========================================
class PackedDatasetLoader:
    def __init__(self, batch_size: int, dataset: list, sequence_length: int, columns: list = None, maps: dict = None, pad_token_id: int = 0):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        
        self.columns = columns
        self.maps = dict([(key, maps[key]) if maps and key in maps.keys() else (key, lambda x: x) for key in self.columns])
        
        self.dataset = self._pack_dataset(dataset)
        
        self.n = 0
        self.max_n = len(self.dataset)

    def _pack_dataset(self, dataset: list) -> list:
        if not dataset:
            return []
            
        base_col = self.columns[0]
        
        # STEP 1: Strict Truncation
        processed_docs = []
        for row in dataset:
            seq = row[base_col].tolist()
            
            # If it's too long, chop the tail off and discard it.
            if len(seq) > self.sequence_length:
                processed_docs.append(seq[:self.sequence_length])
            else:
                processed_docs.append(seq)
                
        packed_bins = pack_best_fit_sorted(processed_docs, capacity=self.sequence_length)
        
        # STEP 3: Pad the dataset and wrap in memory-safe NumPy arrays
        packed_dataset = []
        for b in packed_bins:
            pad_len = self.sequence_length - len(b)
            
            # Pad exactly up to sequence length
            if pad_len > 0:
                b.extend([self.pad_token_id] * pad_len)
                
            # Cast immediately back to NumPy int64 to prevent SegFaults
            padded_array = np.array(b, dtype=np.int64)
            
            packed_row = {base_col: padded_array}
            packed_dataset.append(packed_row)
            
        return packed_dataset

    def __iter__(self):
        return self

    def __next__(self):
        outputs = {key: [] for key in self.columns}
        
        for i in range(self.batch_size):
            if self.n >= self.max_n:
                if i == 0:
                    self.cleanup()
                    raise StopIteration
                break 

            for key in self.columns:
                outputs[key].append(self.maps[key](self.dataset[self.n][key]))
                
            self.n += 1
            
        return {key: np.stack(val) for key, val in outputs.items()}

    def cleanup(self):        
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.dataset = None
        gc.collect()

    def __del__(self):
        self.cleanup()


def load_packed_parquet(base_path, dataset_name, batch_size: int, sequence_length: int, columns: list, maps: dict = None, pad_token_id: int = 0):
    if maps is None:
        maps = {}
        
    full_path = f"{base_path}{dataset_name}"
    table = pq.read_table(full_path, columns=columns)
    
    buffer = []
    for batch in table.to_batches():
        num_rows = batch.num_rows
        for i in range(num_rows):
            row_dict = {}
            for col in columns:
                row_dict[col] = np.array(batch[col][i].as_py(), dtype=np.int64)
            buffer.append(row_dict)

    del table
    gc.collect()

    loader = PackedDatasetLoader(
        batch_size=batch_size, 
        dataset=buffer, 
        sequence_length=sequence_length, 
        columns=columns, 
        maps=maps, 
        pad_token_id=pad_token_id
    )
    
    return loader, len(loader.dataset)
