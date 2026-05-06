#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试toa处理逻辑的差异
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CRASH'))

from src.data_loader import CrashDataset
from torch.utils.data import DataLoader

def process_toa_simple(toa, device):
    """简化版toa处理（AP=0.9827）"""
    if isinstance(toa, list):
        toa_flat = []
        for item in toa:
            if isinstance(item, torch.Tensor):
                val = item.item() if item.numel() == 1 else float(item[0])
            else:
                val = float(item)
            toa_flat.append(val)
        toa_tensor = torch.tensor(toa_flat, dtype=torch.float32, device=device)
    elif isinstance(toa, torch.Tensor):
        toa_tensor = toa.to(device).float()
        if toa_tensor.dim() > 1:
            toa_tensor = toa_tensor.flatten()
    else:
        toa_tensor = torch.tensor([float(toa)], dtype=torch.float32, device=device)
    
    if toa_tensor.dim() == 0:
        toa_tensor = toa_tensor.unsqueeze(0)
    if len(toa_tensor.shape) == 0 or toa_tensor.shape[0] != 8:  # batch_size=8
        toa_tensor = toa_tensor.expand(8)
    
    return toa_tensor

def process_toa_complex(toa, device):
    """复杂版toa处理（与train_best_params.py一致，AP=0.1390）"""
    if isinstance(toa, list):
        if len(toa) > 0 and isinstance(toa[0], list):
            toa_flat = []
            for item in toa:
                if isinstance(item, list) and len(item) > 0:
                    val = item[0]
                else:
                    val = item
                if isinstance(val, torch.Tensor):
                    val = val.item() if val.numel() == 1 else float(val[0])
                else:
                    val = float(val)
                toa_flat.append(val)
            toa = torch.tensor(toa_flat, dtype=torch.float32, device=device)
        else:
            toa_flat = []
            for item in toa:
                if isinstance(item, torch.Tensor):
                    val = item.item() if item.numel() == 1 else float(item[0])
                else:
                    val = float(item)
                toa_flat.append(val)
            toa = torch.tensor(toa_flat, dtype=torch.float32, device=device)
    elif isinstance(toa, np.ndarray):
        toa = torch.from_numpy(toa).float().to(device)
        if toa.dim() > 1:
            toa = toa.flatten()
    elif not isinstance(toa, torch.Tensor):
        try:
            toa = torch.tensor([float(toa)], dtype=torch.float32, device=device)
        except:
            toa = torch.tensor([0.0], dtype=torch.float32, device=device)
    else:
        toa = toa.to(device, non_blocking=True)
        if toa.dim() > 1:
            toa = toa.flatten()
        elif toa.dim() == 0:
            toa = toa.unsqueeze(0)
    
    return toa

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
test_dataset = CrashDataset('/data/sony/LFCRASH/CRASH/data/crash', 'vgg16', phase='test', toTensor=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

print("="*80)
print("调试toa处理逻辑差异")
print("="*80)

for batch_idx, (x, y, toa) in enumerate(test_loader):
    if batch_idx >= 3:
        break
    
    print(f"\nBatch {batch_idx}:")
    print(f"  原始toa类型: {type(toa)}")
    print(f"  原始toa值: {toa}")
    
    # 简化版处理
    toa_simple = process_toa_simple(toa, device)
    print(f"\n  简化版处理结果:")
    print(f"    toa_tensor: {toa_simple.cpu().numpy()}")
    print(f"    shape: {toa_simple.shape}")
    print(f"    dtype: {toa_simple.dtype}")
    
    # 复杂版处理
    toa_complex = process_toa_complex(toa, device)
    print(f"\n  复杂版处理结果:")
    print(f"    toa_tensor: {toa_complex.cpu().numpy()}")
    print(f"    shape: {toa_complex.shape}")
    print(f"    dtype: {toa_complex.dtype}")
    
    # 对比差异
    if toa_simple.shape == toa_complex.shape:
        diff = torch.abs(toa_simple - toa_complex)
        print(f"\n  差异:")
        print(f"    最大差异: {diff.max().item():.6f}")
        print(f"    平均差异: {diff.mean().item():.6f}")
        if diff.max().item() > 0.01:
            print(f"    ⚠️  差异较大！")
            print(f"    简化版: {toa_simple.cpu().numpy()}")
            print(f"    复杂版: {toa_complex.cpu().numpy()}")
    else:
        print(f"\n  ⚠️  shape不同！")
        print(f"    简化版shape: {toa_simple.shape}")
        print(f"    复杂版shape: {toa_complex.shape}")
