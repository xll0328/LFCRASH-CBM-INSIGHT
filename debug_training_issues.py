#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试训练问题：检查数据加载、模型输出、评估函数
"""

import os
import sys
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..', 'CRASH'))

# 导入模块
try:
    from src.data_loader import DADDataset, CrashDataset, A3DDataset
except ImportError:
    # 尝试从CRASH目录导入
    crash_path = os.path.join(current_dir, '..', 'CRASH')
    sys.path.insert(0, os.path.join(crash_path, 'src'))
    from DataLoader import DADDataset, CrashDataset, A3DDataset

try:
    from src.models_gru import LFCRASH_CBM_GRU
except ImportError:
    try:
        from src.Models import LFCRASH_CBM_GRU
    except ImportError:
        from Models import LFCRASH_CBM_GRU

from src.eval_tools import evaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据集参数
DATASET_PARAMS = {
    "dad": {
        "x_dim": 4096, "n_obj": 19, "n_frames": 100, "fps": 20.0,
        "feature": "vgg16", "phase_train": "training", "phase_test": "testing"
    },
    "crash": {
        "x_dim": 4096, "n_obj": 19, "n_frames": 50, "fps": 10.0,
        "feature": "vgg16", "phase_train": "train", "phase_test": "test"
    },
    "a3d": {
        "x_dim": 4096, "n_obj": 19, "n_frames": 100, "fps": 20.0,
        "feature": "vgg16", "phase_train": "train", "phase_test": "test"
    },
}

DATA_ROOT = "/data/sony/LFCRASH/CRASH/data"

def check_data_loading(dataset_name):
    """检查数据加载"""
    logger.info(f"\n{'='*80}")
    logger.info(f"检查数据加载: {dataset_name.upper()}")
    logger.info(f"{'='*80}")
    
    ds_params = DATASET_PARAMS[dataset_name]
    
    # 数据集路径
    if dataset_name == "crash":
        data_path = os.path.join(DATA_ROOT, "crash")
    else:
        data_path = os.path.join(DATA_ROOT, dataset_name.lower())
    
    try:
        # 加载训练集
        if dataset_name == "dad":
            base_train = DADDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
            train_dataset = base_train
        elif dataset_name == "crash":
            train_dataset = CrashDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
        elif dataset_name == "a3d":
            train_dataset = A3DDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
        
        logger.info(f"✓ 训练集加载成功: {len(train_dataset)} 个样本")
        
        # 检查前几个样本
        logger.info(f"\n检查前3个样本:")
        for i in range(min(3, len(train_dataset))):
            try:
                features, labels, toa = train_dataset[i]
                logger.info(f"  样本 {i}:")
                logger.info(f"    features shape: {features.shape if isinstance(features, np.ndarray) else type(features)}")
                logger.info(f"    labels: {labels}")
                logger.info(f"    toa: {toa} (type: {type(toa)})")
                
                # 检查labels
                if isinstance(labels, np.ndarray):
                    if len(labels.shape) == 2:
                        logger.info(f"    ⚠️  labels是2D数组: {labels.shape}")
                        logger.info(f"    labels内容: {labels}")
                    else:
                        logger.info(f"    labels shape: {labels.shape}")
                        logger.info(f"    labels值: {labels}")
                
                # 检查toa
                if isinstance(toa, (list, np.ndarray)):
                    logger.info(f"    ⚠️  toa是列表/数组: {toa}")
                elif isinstance(toa, torch.Tensor):
                    logger.info(f"    ⚠️  toa是Tensor: {toa.shape}, value={toa}")
            except Exception as e:
                logger.error(f"  样本 {i} 加载失败: {e}", exc_info=True)
        
        # 加载测试集
        if dataset_name == "dad":
            base_test = DADDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
            test_dataset = base_test
        elif dataset_name == "crash":
            test_dataset = CrashDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
        elif dataset_name == "a3d":
            test_dataset = A3DDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
        
        logger.info(f"\n✓ 测试集加载成功: {len(test_dataset)} 个样本")
        
        # 检查测试集样本
        logger.info(f"\n检查测试集前3个样本:")
        for i in range(min(3, len(test_dataset))):
            try:
                features, labels, toa = test_dataset[i]
                logger.info(f"  样本 {i}:")
                logger.info(f"    labels: {labels}")
                logger.info(f"    toa: {toa}")
            except Exception as e:
                logger.error(f"  样本 {i} 加载失败: {e}", exc_info=True)
        
        return True
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}", exc_info=True)
        return False

def check_model_output(dataset_name, device='cuda:0'):
    """检查模型输出"""
    logger.info(f"\n{'='*80}")
    logger.info(f"检查模型输出: {dataset_name.upper()}")
    logger.info(f"{'='*80}")
    
    ds_params = DATASET_PARAMS[dataset_name]
    
    # 使用最佳超参数
    BEST_PARAMS = {
        "dad": {
            "lambda_align": 2.4017632837038076e-05,
            "lambda_sparse": 0.0002624009864715832,
            "batch_size": 8,
            "learning_rate": 0.00039865649058389127,
            "weight_decay": 2.778678567292157e-05,
            "h_dim": 256,
            "z_dim": 128,
        },
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
    
    params = BEST_PARAMS[dataset_name]
    
    try:
        # 加载数据集
        if dataset_name == "crash":
            data_path = os.path.join(DATA_ROOT, "crash")
        else:
            data_path = os.path.join(DATA_ROOT, dataset_name.lower())
        
        if dataset_name == "dad":
            base_test = DADDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
            test_dataset = base_test
        elif dataset_name == "crash":
            test_dataset = CrashDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
        elif dataset_name == "a3d":
            test_dataset = A3DDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
        
        test_loader = DataLoader(test_dataset, batch_size=min(params["batch_size"], 4), shuffle=False, num_workers=0)
        
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
        
        model.eval()
        
        logger.info(f"\n检查模型输出（前3个batch）:")
        all_probs = []
        all_labels = []
        all_toas = []
        
        with torch.no_grad():
            for batch_idx, (x, y, toa) in enumerate(test_loader):
                if batch_idx >= 3:
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
                
                # 获取预测概率
                if len(all_outputs) > 0:
                    last_output = all_outputs[-1]  # (B, 2)
                    video_probs = torch.softmax(last_output, dim=-1)[:, 1]  # (B,)
                    
                    logger.info(f"\n  Batch {batch_idx}:")
                    logger.info(f"    输入shape: {x.shape}")
                    logger.info(f"    标签: {y.cpu().numpy()}")
                    logger.info(f"    toa: {toa_tensor.cpu().numpy()}")
                    logger.info(f"    模型输出logits: {last_output.cpu().numpy()}")
                    logger.info(f"    预测概率: {video_probs.cpu().numpy()}")
                    logger.info(f"    预测概率范围: [{video_probs.min().item():.4f}, {video_probs.max().item():.4f}]")
                    logger.info(f"    预测概率均值: {video_probs.mean().item():.4f}")
                    
                    all_probs.append(video_probs.cpu().numpy())
                    all_labels.append(y[:, 1].cpu().numpy() if y.shape[1] > 1 else y.cpu().numpy())
                    toa_np = toa_tensor.cpu().numpy()
                    if toa_np.ndim > 1:
                        toa_np = toa_np.flatten()
                    all_toas.append(toa_np)
                else:
                    logger.warning(f"  Batch {batch_idx}: 模型没有输出")
        
        if all_probs:
            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            all_toas = np.concatenate(all_toas)
            
            logger.info(f"\n总体统计:")
            logger.info(f"  预测概率范围: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
            logger.info(f"  预测概率均值: {all_probs.mean():.4f}")
            logger.info(f"  预测概率标准差: {all_probs.std():.4f}")
            logger.info(f"  标签分布: {np.bincount(all_labels.astype(int))}")
            logger.info(f"  toa范围: [{all_toas.min():.2f}, {all_toas.max():.2f}]")
            
            # 检查预测值是否合理
            if all_probs.min() < 0 or all_probs.max() > 1:
                logger.error(f"  ❌ 预测概率不在[0,1]范围内！")
            elif np.allclose(all_probs, all_probs[0]):
                logger.error(f"  ❌ 所有预测值都相同！模型可能没有学习")
            elif all_probs.std() < 0.01:
                logger.warning(f"  ⚠️  预测值变化很小（std={all_probs.std():.4f}），模型可能没有学习")
            else:
                logger.info(f"  ✓ 预测值分布正常")
        
        return True
        
    except Exception as e:
        logger.error(f"模型输出检查失败: {e}", exc_info=True)
        return False

def check_evaluation_function(dataset_name):
    """检查evaluation函数"""
    logger.info(f"\n{'='*80}")
    logger.info(f"检查evaluation函数: {dataset_name.upper()}")
    logger.info(f"{'='*80}")
    
    ds_params = DATASET_PARAMS[dataset_name]
    
    # 创建一些测试数据
    n_videos = 10
    n_frames = ds_params["n_frames"]
    
    # 测试1: 正常预测
    logger.info(f"\n测试1: 正常预测")
    all_pred = np.random.rand(n_videos, n_frames) * 0.5 + 0.5  # [0.5, 1.0]
    all_labels = np.random.randint(0, 2, n_videos)
    all_toa = np.random.randint(10, n_frames-10, n_videos).astype(float)
    
    try:
        AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toa, fps=ds_params["fps"])
        logger.info(f"  AP: {AP:.4f}")
        logger.info(f"  mTTA: {mTTA:.4f}")
        logger.info(f"  TTA@R80: {TTA_R80:.4f}")
        logger.info(f"  P@R80: {P_R80:.4f}")
        
        if mTTA == 5.0 and TTA_R80 == 5.0:
            logger.warning(f"  ⚠️  TTA值都是5.0，可能是默认值或计算错误")
    except Exception as e:
        logger.error(f"  评估失败: {e}", exc_info=True)
    
    # 测试2: 所有预测值相同（模拟模型未学习）
    logger.info(f"\n测试2: 所有预测值相同")
    all_pred = np.ones((n_videos, n_frames)) * 0.5
    all_labels = np.random.randint(0, 2, n_videos)
    all_toa = np.random.randint(10, n_frames-10, n_videos).astype(float)
    
    try:
        AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toa, fps=ds_params["fps"])
        logger.info(f"  AP: {AP:.4f}")
        logger.info(f"  mTTA: {mTTA:.4f}")
        logger.info(f"  TTA@R80: {TTA_R80:.4f}")
        logger.info(f"  P@R80: {P_R80:.4f}")
    except Exception as e:
        logger.error(f"  评估失败: {e}", exc_info=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='调试训练问题')
    parser.add_argument('--dataset', type=str, required=True, choices=['dad', 'crash', 'a3d'])
    parser.add_argument('--check_data', action='store_true', help='检查数据加载')
    parser.add_argument('--check_model', action='store_true', help='检查模型输出')
    parser.add_argument('--check_eval', action='store_true', help='检查evaluation函数')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    if args.check_data:
        check_data_loading(args.dataset)
    
    if args.check_model:
        check_model_output(args.dataset, device)
    
    if args.check_eval:
        check_evaluation_function(args.dataset)
    
    if not (args.check_data or args.check_model or args.check_eval):
        # 默认检查所有
        check_data_loading(args.dataset)
        check_model_output(args.dataset, device)
        check_evaluation_function(args.dataset)

if __name__ == "__main__":
    main()
