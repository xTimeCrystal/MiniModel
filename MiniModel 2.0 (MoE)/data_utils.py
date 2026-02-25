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
