#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查完整训练时的实际预测值，使用保存的最佳模型
"""

import os
import sys
import torch
import numpy as np
import json

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CRASH'))

from src.data_loader import CrashDataset, A3DDataset
from src.models_gru import LFCRASH_CBM_GRU
from src.eval_tools import evaluation
from torch.utils.data import DataLoader

DATA_ROOT = "/data/sony/LFCRASH/CRASH/data"

def check_predictions_from_saved_model(dataset_name):
    """使用保存的最佳模型检查预测值"""
    print(f"\n{'='*80}")
    print(f"检查{dataset_name.upper()}的预测值（使用保存的最佳模型）")
    print(f"{'='*80}")
    
    # 最佳超参数
    BEST_PARAMS = {
        "crash": {
            "lambda_align": 1.1384241618312619e-05,
            "lambda_sparse": 0.0011274223027372562,
            "batch_size": 8,
            "learning_rate": 0.00020037071372634206,
            "weight_decay": 9.818247569037478e-05,
            "h_dim": 256,
            "z_dim": 512,
        },
        "a3d": {
            "lambda_align": 0.0006584106160121611,
            "lambda_sparse": 0.004835952776465951,
            "batch_size": 32,
            "learning_rate": 2.4658447214487382e-06,
            "weight_decay": 1.2315571723666031e-06,
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
    
    # 加载最佳模型
    checkpoint_path = f"output/full_training_20260114_121308/best_{dataset_name}/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 加载最佳模型 (Epoch {checkpoint.get('epoch', 'unknown')}, AP={checkpoint.get('ap', 'unknown'):.4f})")
    else:
        print(f"⚠️  最佳模型不存在: {checkpoint_path}")
        print("  使用随机初始化的模型")
    
    model.eval()
    
    all_probs = []
    all_labels = []
    all_toas = []
    
    print(f"\n收集预测值（前100个样本）...")
    with torch.no_grad():
        for batch_idx, (x, y, toa) in enumerate(test_loader):
            if len(all_probs) >= 100:
                break
            
            x = x.to(device).float()
            y = y.to(device).float()
            
            # 处理toa
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
            if len(toa_tensor.shape) == 0 or toa_tensor.shape[0] != x.shape[0]:
                toa_tensor = toa_tensor.expand(x.shape[0])
            
            # 前向传播
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
        
        print(f"\n预测值统计:")
        print(f"  样本数: {len(all_probs)}")
        print(f"  预测概率范围: [{all_probs.min():.6f}, {all_probs.max():.6f}]")
        print(f"  预测概率均值: {all_probs.mean():.6f}")
        print(f"  预测概率标准差: {all_probs.std():.6f}")
        print(f"  预测概率中位数: {np.median(all_probs):.6f}")
        print(f"  唯一值数量: {len(np.unique(all_probs))}")
        
        if len(np.unique(all_probs)) < 10:
            print(f"  ⚠️  警告: 唯一值数量很少，预测值可能都相同或非常接近")
            print(f"  唯一值: {np.unique(all_probs)[:20]}")
        
        print(f"\n标签统计:")
        print(f"  标签分布: {np.bincount(all_labels.astype(int))}")
        print(f"  正样本比例: {np.sum(all_labels) / len(all_labels):.4f}")
        
        print(f"\ntoa统计:")
        print(f"  toa范围: [{all_toas.min():.2f}, {all_toas.max():.2f}]")
        print(f"  toa均值: {all_toas.mean():.2f}")
        
        # 使用evaluation函数计算指标
        print(f"\n计算评估指标...")
        n_frames = ds_params["n_frames"]
        all_pred_frame = np.tile(all_probs[:, None], (1, n_frames))
        
        try:
            AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred_frame, all_labels, all_toas, fps=ds_params["fps"])
            print(f"  AP: {AP:.4f}")
            print(f"  mTTA: {mTTA:.4f}")
            print(f"  TTA@R80: {TTA_R80:.4f}")
            print(f"  P@R80: {P_R80:.4f}")
        except Exception as e:
            print(f"  ❌ 评估失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['crash', 'a3d'])
    args = parser.parse_args()
    
    check_predictions_from_saved_model(args.dataset)
