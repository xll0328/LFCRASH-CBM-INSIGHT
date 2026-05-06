#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optuna离散超参数优化脚本
使用离散超参数空间，提高可解释性和可复现性
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from src.models_gru import LFCRASH_CBM_GRU
from src.data_loader import DADDataset, CrashDataset, A3DDataset
from src.eval_tools import evaluation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# 数据根目录
DATA_ROOT = "/data/sony/LFCRASH/CRASH/data"


def suggest_hyperparameters_discrete(trial, dataset_name):
    """为给定数据集建议离散超参数"""
    if dataset_name == "dad":
        return {
            "lambda_align": trial.suggest_categorical("lambda_align", [
                1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4
            ]),
            "lambda_sparse": trial.suggest_categorical("lambda_sparse", [
                1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3
            ]),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
            "learning_rate": trial.suggest_categorical("learning_rate", [
                1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3
            ]),
            "weight_decay": trial.suggest_categorical("weight_decay", [
                1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4
            ]),
            "h_dim": trial.suggest_categorical("h_dim", [128, 256, 512]),
            "z_dim": trial.suggest_categorical("z_dim", [64, 128, 256]),
        }
    elif dataset_name == "crash":
        return {
            "lambda_align": trial.suggest_categorical("lambda_align", [
                5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4
            ]),
            "lambda_sparse": trial.suggest_categorical("lambda_sparse", [
                1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2
            ]),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
            "learning_rate": trial.suggest_categorical("learning_rate", [
                5e-5, 1e-4, 2e-4, 5e-4, 1e-3
            ]),
            "weight_decay": trial.suggest_categorical("weight_decay", [
                1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4
            ]),
            "h_dim": trial.suggest_categorical("h_dim", [256, 512, 768]),
            "z_dim": trial.suggest_categorical("z_dim", [128, 256, 512, 768]),
        }
    else:  # a3d
        return {
            "lambda_align": trial.suggest_categorical("lambda_align", [
                1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3
            ]),
            "lambda_sparse": trial.suggest_categorical("lambda_sparse", [
                1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2
            ]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "learning_rate": trial.suggest_categorical("learning_rate", [
                1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4
            ]),
            "weight_decay": trial.suggest_categorical("weight_decay", [
                1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5
            ]),
            "h_dim": trial.suggest_categorical("h_dim", [512, 768, 1024]),
            "z_dim": trial.suggest_categorical("z_dim", [128, 256, 512]),
        }


def train_and_evaluate(trial, dataset_name, device, n_epochs=50, use_test_set=False):
    """训练并评估模型，返回测试集AP（如果use_test_set=True）或验证集AP"""
    
    # 获取超参数
    params = suggest_hyperparameters_discrete(trial, dataset_name)
    ds_params = DATASET_PARAMS[dataset_name]
    
    # 加载数据集
    if dataset_name == "crash":
        data_path = os.path.join(DATA_ROOT, "crash")
    else:
        data_path = os.path.join(DATA_ROOT, dataset_name.lower())
    
    if dataset_name == "dad":
        # DAD使用特殊的wrapper处理
        class DADDatasetWrapper(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                try:
                    features, labels, toa = self.dataset[idx]
                except ValueError:
                    data_file = os.path.join(self.dataset.data_path, self.dataset.phase, self.dataset.files_list[idx])
                    data = np.load(data_file)
                    features = data['data']
                    labels = data['labels']
                    has_positive = np.any(labels[:, 1] > 0) if labels.shape[1] > 1 else np.any(labels > 0)
                    video_label = float(has_positive)
                    labels = np.array([1 - video_label, video_label], dtype=np.float32)
                    toa = [90.0] if has_positive else [self.dataset.n_frames + 1]
                if isinstance(toa, (list, np.ndarray)):
                    toa = float(toa[0]) if len(toa) > 0 else float(toa)
                elif isinstance(toa, torch.Tensor):
                    toa = float(toa.item() if toa.numel() == 1 else toa[0])
                return features, labels, np.array([toa], dtype=np.float32)
        
        base_train = DADDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
        train_dataset = DADDatasetWrapper(base_train)
        if use_test_set:
            base_test = DADDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
            eval_dataset = DADDatasetWrapper(base_test)
        else:
            train_size = int(0.8 * len(train_dataset))
            eval_dataset = torch.utils.data.Subset(train_dataset, range(train_size, len(train_dataset)))
            train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    elif dataset_name == "crash":
        train_dataset = CrashDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
        if use_test_set:
            eval_dataset = CrashDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
        else:
            train_size = int(0.8 * len(train_dataset))
            eval_dataset = torch.utils.data.Subset(train_dataset, range(train_size, len(train_dataset)))
            train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    else:  # a3d
        train_dataset = A3DDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
        if use_test_set:
            eval_dataset = A3DDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
        else:
            train_size = int(0.8 * len(train_dataset))
            eval_dataset = torch.utils.data.Subset(train_dataset, range(train_size, len(train_dataset)))
            train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    
    # 创建DataLoader
    if dataset_name == "dad":
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0)
        eval_loader = DataLoader(eval_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # 初始化模型
    concept_file = os.path.join(os.path.dirname(__file__), "..", "000_all_concept_set.txt")
    if not os.path.exists(concept_file):
        concept_file = None
    
    model = LFCRASH_CBM_GRU(
        x_dim=ds_params["x_dim"],
        h_dim=params["h_dim"],
        z_dim=params["z_dim"],
        n_layers=2,
        n_obj=ds_params["n_obj"],
        n_frames=ds_params["n_frames"],
        fps=ds_params["fps"],
        with_saa=True,
        num_concepts=837,
        concept_file=concept_file,
        lambda_align=params["lambda_align"],
        lambda_sparse=params["lambda_sparse"],
        device=device
    ).to(device)
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"]
    )
    
    best_ap = 0.0
    
    # 训练循环
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, (x, y, toa) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            
            # 处理toa
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
                    toa_tensor = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                else:
                    toa_flat = []
                    for item in toa:
                        if isinstance(item, torch.Tensor):
                            val = item.item() if item.numel() == 1 else float(item[0])
                        else:
                            val = float(item)
                        toa_flat.append(val)
                    toa_tensor = torch.tensor(toa_flat, dtype=torch.float32, device=device)
            else:
                if isinstance(toa, torch.Tensor):
                    if toa.dim() > 1:
                        toa_tensor = toa.squeeze().to(device).float()
                    else:
                        toa_tensor = toa.to(device).float()
                else:
                    toa_tensor = torch.tensor(toa, dtype=torch.float32, device=device)
            
            toa_tensor = toa_tensor.to(device)
            if toa_tensor.dim() == 0:
                toa_tensor = toa_tensor.unsqueeze(0)
            if len(toa_tensor.shape) == 0 or toa_tensor.shape[0] != x.shape[0]:
                toa_tensor = toa_tensor.expand(x.shape[0])
            
            optimizer.zero_grad()
            
            # 前向传播
            losses, _, _ = model(x, y, toa_tensor)
            total_loss = losses['total_loss']
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
        
        # 每个epoch后评估
        if epoch % 5 == 0 or epoch == n_epochs:
            model.eval()
            all_pred_frame_list = []  # 收集frame-level predictions
            all_labels = []
            all_toa = []
            
            with torch.no_grad():
                for x, y, toa in eval_loader:
                    batch_size = x.size(0)
                    x = x.to(device, non_blocking=True).float()
                    y = y.to(device, non_blocking=True).float()
                    
                    # 处理toa
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
                            toa_tensor = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                        else:
                            toa_flat = []
                            for item in toa:
                                if isinstance(item, torch.Tensor):
                                    val = item.item() if item.numel() == 1 else float(item[0])
                                else:
                                    val = float(item)
                                toa_flat.append(val)
                            toa_tensor = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                    else:
                        if isinstance(toa, torch.Tensor):
                            if toa.dim() > 1:
                                toa_tensor = toa.squeeze().to(device).float()
                            else:
                                toa_tensor = toa.to(device).float()
                        else:
                            toa_tensor = torch.tensor(toa, dtype=torch.float32, device=device)
                    
                    toa_tensor = toa_tensor.to(device)
                    if toa_tensor.dim() == 0:
                        toa_tensor = toa_tensor.unsqueeze(0)
                    if len(toa_tensor.shape) == 0 or toa_tensor.shape[0] != batch_size:
                        toa_tensor = toa_tensor.expand(batch_size)
                    
                    _, all_outputs, _ = model(x, None, toa_tensor)
                    
                    # 收集frame-level predictions（修复TTA问题）
                    for frame_idx in range(len(all_outputs)):
                        frame_output = all_outputs[frame_idx]  # (B, 2)
                        frame_probs = torch.softmax(frame_output, dim=-1)[:, 1]  # (B,)
                        if len(all_pred_frame_list) <= frame_idx:
                            all_pred_frame_list.append([])
                        all_pred_frame_list[frame_idx].extend(frame_probs.cpu().numpy())
                    
                    # 收集标签和toa
                    all_labels.extend(y[:, 1].cpu().numpy() if y.shape[1] > 1 else y.cpu().numpy())
                    toa_np = toa_tensor.cpu().numpy()
                    if toa_np.ndim > 1:
                        toa_np = toa_np.flatten()
                    all_toa.extend(toa_np)
            
            # 计算AP
            if len(all_pred_frame_list) > 0 and len(all_labels) > 0:
                # 转置frame-level predictions: (n_frames, n_videos) -> (n_videos, n_frames)
                all_pred = np.array(all_pred_frame_list).T  # (n_videos, n_frames)
                all_labels = np.array(all_labels)
                all_toa = np.array(all_toa)
                
                ap, mtta, tta_r80, p_r80 = evaluation(
                    all_pred, all_labels, all_toa, fps=ds_params["fps"]
                )
                
                # 报告中间值给Optuna（用于剪枝）
                trial.report(ap, epoch)
                
                # 检查是否应该剪枝
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                if ap > best_ap:
                    best_ap = ap
                
                # 记录最佳结果
                trial.set_user_attr("best_ap", best_ap)
                trial.set_user_attr("best_epoch", epoch)
    
    return best_ap


def objective(trial, dataset_name, device, n_epochs, use_test_set):
    """Optuna目标函数"""
    try:
        ap = train_and_evaluate(trial, dataset_name, device, n_epochs, use_test_set)
        return ap
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial失败: {e}", exc_info=True)
        return 0.0  # 返回最差分数


def main():
    parser = argparse.ArgumentParser(description='Optuna离散超参数优化')
    parser.add_argument('--dataset', type=str, required=True, choices=['dad', 'crash', 'a3d'],
                        help='要优化的数据集')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--n_trials', type=int, default=10, help='试验次数（默认10）')
    parser.add_argument('--n_epochs', type=int, default=50, help='每个trial的epoch数')
    parser.add_argument('--use_test_set', action='store_true',
                        help='使用测试集进行优化（用户允许）')
    parser.add_argument('--output_dir', type=str, default='output/optuna_discrete',
                        help='输出目录')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Study名称（用于恢复）')
    
    args = parser.parse_args()
    
    # 设置设备
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
        device = torch.device(f'cuda:0')
        logger.info(f"检测到CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}，使用cuda:0")
    else:
        device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"使用设备: {device}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"试验次数: {args.n_trials}")
    logger.info(f"每个trial的epoch数: {args.n_epochs}")
    logger.info(f"使用测试集优化: {args.use_test_set}")
    logger.info("=" * 80)
    logger.info("使用离散超参数空间")
    logger.info("=" * 80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Study名称
    if args.study_name is None:
        args.study_name = f"optuna_discrete_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=5)  # 对于10个trials，减少startup
    )
    
    # 运行优化
    logger.info("开始Optuna离散超参数优化...")
    study.optimize(
        lambda trial: objective(trial, args.dataset, device, args.n_epochs, args.use_test_set),
        n_trials=args.n_trials,
        timeout=None,
        show_progress_bar=True
    )
    
    # 保存结果
    best_trial = study.best_trial
    logger.info("=" * 80)
    logger.info(f"优化完成！最佳AP: {best_trial.value:.4f}")
    logger.info(f"最佳参数:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
    
    # 保存结果到JSON
    results = {
        "dataset": args.dataset,
        "best_ap": best_trial.value,
        "best_params": best_trial.params,
        "best_trial_number": best_trial.number,
        "n_trials": len(study.trials),
        "use_test_set": args.use_test_set,
        "n_epochs": args.n_epochs,
        "hyperparameter_space": "discrete"
    }
    
    results_file = os.path.join(args.output_dir, f"{args.dataset}_discrete_optuna_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"结果已保存到: {results_file}")
    
    # 保存study对象
    study_file = os.path.join(args.output_dir, f"{args.dataset}_discrete_optuna_study.pkl")
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)
    
    logger.info(f"Study对象已保存到: {study_file}")
    
    # 打印所有trials的结果
    logger.info("\n所有Trials结果:")
    logger.info("-" * 80)
    for i, trial in enumerate(study.trials):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(f"Trial {i}: AP={trial.value:.4f}, Params={trial.params}")


if __name__ == '__main__':
    main()
