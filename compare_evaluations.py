#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比训练时评估和重新评估的差异
"""

import os
import sys
import torch
import numpy as np
import re

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CRASH'))

from src.data_loader import CrashDataset, A3DDataset
from src.models_gru import LFCRASH_CBM_GRU
from src.eval_tools import evaluation
from torch.utils.data import DataLoader

DATA_ROOT = "/data/sony/LFCRASH/CRASH/data"

def extract_training_evaluation_data(dataset_name, timestamp="20260114_121308"):
    """从训练日志中提取评估时的数据信息"""
    log_file = f"logs/full_training_{dataset_name}_{timestamp}.log"
    
    if not os.path.exists(log_file):
        return None
    
    # 提取第1个epoch的评估结果
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 查找Epoch 1的评估结果
    pattern = r'Epoch \[1/\d+\].*?AP \(video-level\): ([\d.]+).*?mTTA: ([\d.]+)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return {
            'ap': float(match.group(1)),
            'mtta': float(match.group(2))
        }
    return None

def compare_evaluations(dataset_name):
    """对比训练时和重新评估的结果"""
    print(f"\n{'='*80}")
    print(f"对比{dataset_name.upper()}的评估结果")
    print(f"{'='*80}")
    
    # 从日志中提取训练时的评估结果
    training_result = extract_training_evaluation_data(dataset_name)
    if training_result:
        print(f"\n训练时评估结果 (Epoch 1):")
        print(f"  AP: {training_result['ap']:.4f}")
        print(f"  mTTA: {training_result['mtta']:.4f}")
    else:
        print(f"\n⚠️  无法从日志中提取训练时的评估结果")
    
    # 重新评估
    BEST_PARAMS = {
        "crash": {
            "lambda_align": 1.1384241618312619e-05,
            "lambda_sparse": 0.0011274223027372562,
            "batch_size": 8,
            "h_dim": 256,
            "z_dim": 512,
        },
        "a3d": {
            "lambda_align": 0.0006584106160121611,
            "lambda_sparse": 0.004835952776465951,
            "batch_size": 32,
            "h_dim": 768,
            "z_dim": 128,
        }
    }
    
    DATASET_PARAMS = {
        "crash": {"x_dim": 4096, "n_obj": 19, "n_frames": 50, "fps": 10.0, "phase_test": "test"},
        "a3d": {"x_dim": 4096, "n_obj": 19, "n_frames": 100, "fps": 20.0, "phase_test": "test"}
    }
    
    params = BEST_PARAMS[dataset_name]
    ds_params = DATASET_PARAMS[dataset_name]
    
    # 加载数据集
    if dataset_name == "crash":
        data_path = os.path.join(DATA_ROOT, "crash")
        test_dataset = CrashDataset(data_path, "vgg16", phase=ds_params["phase_test"], toTensor=False)
    elif dataset_name == "a3d":
        data_path = os.path.join(DATA_ROOT, "a3d")
        test_dataset = A3DDataset(data_path, "vgg16", phase=ds_params["phase_test"], toTensor=False)
    
    # 使用与训练时相同的batch_size
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 初始化模型
    concept_file = os.path.join(os.path.dirname(__file__), "..", "000_all_concept_set.txt")
    if not os.path.exists(concept_file):
        concept_file = None
    
    model = LFCRASH_CBM_GRU(
        x_dim=ds_params["x_dim"],
        n_obj=ds_params["n_obj"],
        h_dim=params["h_dim"],
        z_dim=params["z_dim"],
        concept_file=concept_file,
        lambda_align=params["lambda_align"],
        lambda_sparse=params["lambda_sparse"],
        device=device
    ).to(device)
    
    # 加载Epoch 1的模型（如果存在）
    checkpoint_path = f"output/full_training_20260114_121308/best_{dataset_name}/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✓ 加载模型 (Epoch {checkpoint.get('epoch', 'unknown')})")
    
    model.eval()
    
    all_probs = []
    all_labels = []
    all_toas = []
    
    print(f"\n重新评估（使用完整测试集）...")
    with torch.no_grad():
        for x, y, toa in test_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            
            # 处理toa（与train_best_params.py保持一致）
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
            
            batch_size = x.size(0)
            toa_tensor = toa
            
            _, all_outputs, _ = model(x, None, toa_tensor)
            
            if len(all_outputs) > 0:
                last_output = all_outputs[-1]
                video_probs = torch.softmax(last_output, dim=-1)[:, 1]
                
                all_probs.append(video_probs.cpu().numpy())
                all_labels.append(y[:, 1].cpu().numpy() if y.shape[1] > 1 else y.cpu().numpy())
                toa_np = toa_tensor.cpu().numpy()
                if toa_np.ndim > 1:
                    toa_np = toa_np.flatten()
                all_toas.append(toa_np)
    
    if all_probs:
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        all_toas = np.concatenate(all_toas)
        if all_toas.ndim > 1:
            all_toas = all_toas.flatten()
        
        print(f"\n重新评估结果:")
        print(f"  样本数: {len(all_probs)}")
        print(f"  预测概率范围: [{all_probs.min():.6f}, {all_probs.max():.6f}]")
        print(f"  预测概率均值: {all_probs.mean():.6f}")
        print(f"  标签分布: {np.bincount(all_labels.astype(int))}")
        
        n_frames = ds_params["n_frames"]
        all_pred = np.tile(all_probs[:, None], (1, n_frames))
        
        AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toas, fps=ds_params["fps"])
        
        print(f"\n最终指标:")
        print(f"  AP: {AP:.4f}")
        print(f"  mTTA: {mTTA:.4f}")
        print(f"  TTA@R80: {TTA_R80:.4f}")
        print(f"  P@R80: {P_R80:.4f}")
        
        if training_result:
            print(f"\n对比:")
            print(f"  训练时AP: {training_result['ap']:.4f}")
            print(f"  重新评估AP: {AP:.4f}")
            print(f"  差异: {AP - training_result['ap']:.4f}")
            if abs(AP - training_result['ap']) > 0.1:
                print(f"  ⚠️  差异很大！可能的原因:")
                print(f"    1. 训练时评估使用的模型状态不同")
                print(f"    2. 训练时评估的数据处理有问题")
                print(f"    3. 训练时评估的代码有bug")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['crash', 'a3d'])
    args = parser.parse_args()
    
    compare_evaluations(args.dataset)
