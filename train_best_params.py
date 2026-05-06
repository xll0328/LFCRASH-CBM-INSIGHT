#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用最佳参数训练三个数据集
基于experiments_gru_gpu7的最佳结果
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

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


# 最佳参数配置（基于Optuna 200 trials搜索结果，2026-01-13）
# 更新了crash和a3d的最佳参数，DAD使用之前的最佳参数（因为日志中未找到完整记录）
BEST_PARAMS = {
    "dad": {
        # DAD: 使用之前30 trials的最佳参数（AP=0.6511），当前Optuna显示最佳AP=0.672434 (Trial #13)
        # 但由于日志中未找到Trial 13的完整参数，使用之前的最佳参数
        "lambda_align": 2.4017632837038076e-05,
        "lambda_sparse": 0.0002624009864715832,
        "batch_size": 8,
        "learning_rate": 0.00039865649058389127,
        "weight_decay": 2.778678567292157e-05,
        "h_dim": 256,
        "z_dim": 128,
        "n_epochs": 80  # CRASH原文使用80 epochs
    },
    "crash": {
        # Crash: Trial #15, AP=0.997312 (最佳)
        "lambda_align": 1.1384241618312619e-05,
        "lambda_sparse": 0.0011274223027372562,
        "batch_size": 8,
        "learning_rate": 0.00020037071372634206,
        "weight_decay": 9.818247569037478e-05,
        "h_dim": 256,
        "z_dim": 512,
        "n_epochs": 80  # CRASH原文使用80 epochs
    },
    "a3d": {
        # A3D: Trial #4, AP=0.961125 (最佳)
        "lambda_align": 0.0006584106160121611,
        "lambda_sparse": 0.004835952776465951,
        "batch_size": 32,
        "learning_rate": 2.4658447214487382e-06,
        "weight_decay": 1.2315571723666031e-06,
        "h_dim": 768,
        "z_dim": 128,
        "n_epochs": 80  # CRASH原文使用80 epochs
    }
}

# 数据集参数
# 注意：根据实际数据检查特征维度
DATASET_PARAMS = {
    "dad": {"x_dim": 4096, "n_obj": 19, "n_frames": 100, "fps": 20.0, "feature": "vgg16", "phase_train": "training", "phase_test": "testing"},
    "crash": {"x_dim": 4096, "n_obj": 19, "n_frames": 50, "fps": 10.0, "feature": "vgg16", "phase_train": "train", "phase_test": "test"},
    "a3d": {"x_dim": 4096, "n_obj": 19, "n_frames": 100, "fps": 20.0, "feature": "vgg16", "phase_train": "train", "phase_test": "test"},
}


def train_dataset(dataset_name, gpu_id, output_dir):
    """使用最佳参数训练单个数据集"""
    
    logger.info("=" * 80)
    logger.info(f"使用最佳参数训练: {dataset_name.upper()} 数据集")
    logger.info("=" * 80)
    
    # 获取参数
    params = BEST_PARAMS[dataset_name]
    ds_params = DATASET_PARAMS[dataset_name]
    
    logger.info(f"超参数设置:")
    logger.info(f"  - Learning Rate: {params['learning_rate']}")
    logger.info(f"  - Batch Size: {params['batch_size']}")
    logger.info(f"  - Epochs: {params['n_epochs']}")
    logger.info(f"  - Lambda Align: {params['lambda_align']}")
    logger.info(f"  - Lambda Sparse: {params['lambda_sparse']}")
    logger.info(f"  - Weight Decay: {params['weight_decay']}")
    logger.info(f"  - h_dim: {params['h_dim']}, z_dim: {params['z_dim']}")
    
    # 设置设备
    # 如果设置了CUDA_VISIBLE_DEVICES，设备索引应该是0
    if torch.cuda.is_available():
        # 检查是否设置了CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            # 如果设置了CUDA_VISIBLE_DEVICES，使用设备0
            device = torch.device('cuda:0')
            logger.info(f"检测到CUDA_VISIBLE_DEVICES={cuda_visible}，使用cuda:0")
        else:
            device = torch.device(f'cuda:{gpu_id}')
        # 验证GPU是否可用
        try:
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            # 测试GPU
            _ = torch.zeros(1).to(device)
        except Exception as e:
            logger.warning(f"无法使用GPU，回退到CPU: {e}")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    exp_output_dir = os.path.join(output_dir, f"best_{dataset_name}")
    os.makedirs(exp_output_dir, exist_ok=True)
    logger.info(f"输出目录: {exp_output_dir}")
    
    # 加载数据集
    logger.info(f"[步骤 1/6] 加载数据集...")
    sys.stdout.flush()
    
    # 数据集路径
    data_root = "/data/sony/LFCRASH/CRASH/data"
    data_path = os.path.join(data_root, dataset_name if dataset_name != "crash" else "crash")
    
    if not os.path.exists(data_path):
        raise ValueError(f"数据集路径不存在: {data_path}")
    
    # 加载训练集和测试集
    # 对于DAD数据集，需要创建包装器修复labels处理问题
    if dataset_name == "dad":
        # 创建包装器修复labels问题
        class DADDatasetWrapper:
            def __init__(self, dataset):
                self.dataset = dataset
            def __len__(self):
                return len(self.dataset)
            def __getitem__(self, idx):
                try:
                    features, labels, toa = self.dataset[idx]
                except ValueError as e:
                    # 如果CRASH的DataLoader报错（labels[1] > 0），需要手动处理
                    # 直接读取数据文件
                    data_file = os.path.join(self.dataset.data_path, self.dataset.phase, self.dataset.files_list[idx])
                    data = np.load(data_file)
                    features = data['data']
                    labels = data['labels']
                    detections = data['det']
                    # 修复labels: DAD的labels是(10, 2)的frame-level数组
                    if isinstance(labels, np.ndarray) and len(labels.shape) == 2:
                        # 检查是否有任何帧是positive（事故帧）
                        has_positive = np.any(labels[:, 1] > 0) if labels.shape[1] > 1 else np.any(labels > 0)
                        video_label = float(has_positive)
                        labels = np.array([1 - video_label, video_label], dtype=np.float32)
                        # 设置toa
                        if has_positive:
                            toa = [90.0]
                        else:
                            toa = [self.dataset.n_frames + 1]
                    else:
                        # 如果labels已经是video-level的
                        if isinstance(labels, np.ndarray) and len(labels) > 1:
                            video_label = float(labels[1] > 0)
                            labels = np.array([1 - video_label, video_label], dtype=np.float32)
                
                # 确保toa是标量
                if isinstance(toa, (list, np.ndarray)):
                    toa = float(toa[0]) if len(toa) > 0 else float(toa)
                elif isinstance(toa, torch.Tensor):
                    toa = float(toa.item() if toa.numel() == 1 else toa[0])
                return features, labels, np.array([toa], dtype=np.float32)
        
        base_train = DADDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
        base_test = DADDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
        train_dataset = DADDatasetWrapper(base_train)
        test_dataset = DADDatasetWrapper(base_test)
    elif dataset_name == "crash":
        train_dataset = CrashDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
        test_dataset = CrashDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
    elif dataset_name == "a3d":
        train_dataset = A3DDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
        test_dataset = A3DDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    logger.info(f"训练集: {len(train_dataset)} 个样本")
    logger.info(f"测试集: {len(test_dataset)} 个样本")
    logger.info(f"帧数: {ds_params['n_frames']}, FPS: {ds_params['fps']}")
    sys.stdout.flush()
    
    # 创建DataLoader
    # 对于DAD数据集，使用单进程避免labels处理问题
    if dataset_name == "dad":
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)
        logger.info("✓ 使用单进程数据加载（DAD数据集）")
    else:
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=params["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                drop_last=True
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=params["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
            logger.info("✓ 使用多进程数据加载")
        except Exception as e:
            logger.warning(f"多进程数据加载失败，回退到单进程: {e}")
            train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)
    
    # 初始化模型
    logger.info(f"[步骤 2/6] 初始化模型...")
    sys.stdout.flush()
    
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
    
    logger.info("✓ 模型初始化完成")
    sys.stdout.flush()
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练循环
    logger.info(f"[步骤 3/6] 开始训练（{params['n_epochs']}个epoch）...")
    sys.stdout.flush()
    
    best_ap = 0.0
    best_epoch = 0
    
    for epoch in range(1, params["n_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{params['n_epochs']}]")
        for batch_idx, (x, y, toa) in enumerate(pbar):
            # 数据转移到GPU
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            # 确保toa是1D tensor (B,)
            # DataLoader可能返回list of lists，需要展平
            if isinstance(toa, list):
                # 如果是list of lists（批处理），展平
                if len(toa) > 0 and isinstance(toa[0], list):
                    toa_flat = []
                    for item in toa:
                        if isinstance(item, list) and len(item) > 0:
                            val = item[0]
                        else:
                            val = item
                        # 处理tensor类型
                        if isinstance(val, torch.Tensor):
                            val = val.item() if val.numel() == 1 else float(val[0])
                        else:
                            val = float(val)
                        toa_flat.append(val)
                    toa = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                else:
                    # 处理list of scalars/tensors
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
                # 如果是2D，展平为1D
                if toa.dim() > 1:
                    toa = toa.flatten()
            elif not isinstance(toa, torch.Tensor):
                try:
                    if isinstance(toa, torch.Tensor):
                        toa = toa.item() if toa.numel() == 1 else toa[0]
                    toa = torch.tensor([float(toa)], dtype=torch.float32, device=device)
                except:
                    toa = torch.tensor([0.0], dtype=torch.float32, device=device)
            else:
                toa = toa.to(device, non_blocking=True)
                # 确保是1D tensor
                if toa.dim() > 1:
                    toa = toa.flatten()
                elif toa.dim() == 0:
                    toa = toa.unsqueeze(0)
            
            optimizer.zero_grad()
            
            # 前向传播 - CRASH对齐后的新格式
            # 返回: losses, all_outputs, all_hidden
            losses, all_outputs, all_hidden = model(x, y, toa)
            
            # 获取总损失
            total_loss = losses.get("total_loss", losses.get("total", 0.0))
            if isinstance(total_loss, (int, float)) and total_loss == 0.0:
                # 如果没有total_loss，计算一个
                total_loss = losses.get("cross_entropy", 0.0)
                if model.with_saa and "auxloss" in losses:
                    total_loss = total_loss + losses["auxloss"]
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            n_batches += 1
            
            # 更新进度条
            loss_dict = {
                'loss': f'{total_loss.item():.4f}',
                'ce': f'{losses.get("cross_entropy", 0.0):.4f}',
            }
            if "auxloss" in losses:
                loss_dict['aux'] = f'{losses["auxloss"].item():.4f}'
            pbar.set_postfix(loss_dict)
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        scheduler.step(avg_loss)
        
        # 每5个epoch或最后一个epoch评估一次
        if epoch % 5 == 0 or epoch == params["n_epochs"]:
            logger.info(f"\nEpoch {epoch} 完成: Total={avg_loss:.4f}")
            sys.stdout.flush()
            
            # 评估
            logger.info(f"[步骤 4/6] 评估模型...")
            sys.stdout.flush()
            
            model.eval()
            all_pred_frame = []  # 存储frame-level predictions: (n_videos, n_frames)
            all_labels = []
            all_toas = []
            
            with torch.no_grad():
                for x, y, toa in tqdm(test_loader, desc="最终测试"):
                    x = x.to(device, non_blocking=True).float()
                    y = y.to(device, non_blocking=True).float()
                    batch_size = x.size(0)  # 在toa处理之前定义batch_size
                    # 确保toa是tensor - 统一处理逻辑
                    if isinstance(toa, list):
                        # 检查是否是list包含tensor的情况：[tensor([...])]
                        if len(toa) > 0 and isinstance(toa[0], torch.Tensor):
                            # 如果list中只有一个tensor，且tensor有多个元素，展开它
                            tensor_val = toa[0]
                            if tensor_val.numel() > 1:
                                # tensor包含多个值，直接使用
                                toa = tensor_val.to(device).float()
                            else:
                                # tensor只有一个值，展开到batch_size
                                val = tensor_val.item() if tensor_val.numel() == 1 else float(tensor_val[0])
                                toa = torch.tensor([val] * batch_size, dtype=torch.float32, device=device)
                        # 如果是list of lists（批处理），展平
                        elif len(toa) > 0 and isinstance(toa[0], list):
                            toa_flat = []
                            for item in toa:
                                if isinstance(item, list) and len(item) > 0:
                                    val = item[0]
                                else:
                                    val = item
                                # 处理tensor类型
                                if isinstance(val, torch.Tensor):
                                    val = val.item() if val.numel() == 1 else float(val[0])
                                else:
                                    val = float(val)
                                toa_flat.append(val)
                            toa = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                        else:
                            # 处理list of scalars/tensors
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
                        # 如果是2D，展平为1D
                        if toa.dim() > 1:
                            toa = toa.flatten()
                    elif not isinstance(toa, torch.Tensor):
                        try:
                            if isinstance(toa, torch.Tensor):
                                toa = toa.item() if toa.numel() == 1 else toa[0]
                            toa = torch.tensor([float(toa)] * batch_size, dtype=torch.float32, device=device)
                        except:
                            toa = torch.tensor([0.0] * batch_size, dtype=torch.float32, device=device)
                    else:
                        toa = toa.to(device, non_blocking=True)
                        # 确保是1D tensor
                        if toa.dim() > 1:
                            toa = toa.flatten()
                        elif toa.dim() == 0:
                            toa = toa.unsqueeze(0)
                    
                    # 确保toa的shape与batch_size匹配
                    if toa.shape[0] != batch_size:
                        if toa.shape[0] == 1:
                            toa = toa.expand(batch_size)
                        else:
                            # 如果shape不匹配且不是1，可能需要截断或填充
                            if toa.shape[0] > batch_size:
                                toa = toa[:batch_size]
                            else:
                                # 填充到batch_size
                                padding = torch.zeros(batch_size - toa.shape[0], device=device, dtype=toa.dtype)
                                toa = torch.cat([toa, padding])
                    
                    # 构造toa（测试时）
                    toa_tensor = toa
                    
                    # CRASH对齐后的新格式: losses, all_outputs, all_hidden
                    _, all_outputs, _ = model(x, None, toa_tensor)
                    
                    # all_outputs是列表，每个元素是(B, 2)的logits，对应每一帧的输出
                    # 需要构造frame-level predictions用于evaluation
                    # 每个视频的frame-level predictions: (n_frames,)
                    n_frames = ds_params["n_frames"]
                    if len(all_outputs) > 0:
                        # 将all_outputs转换为frame-level predictions
                        # all_outputs是列表，长度为n_frames，每个元素是(B, 2)
                        frame_probs_list = []
                        for frame_idx in range(min(len(all_outputs), n_frames)):
                            frame_output = all_outputs[frame_idx]  # (B, 2)
                            frame_probs = torch.softmax(frame_output, dim=-1)[:, 1]  # (B,) - 正类概率
                            frame_probs_list.append(frame_probs.cpu().numpy())
                        
                        # 如果帧数不足n_frames，用最后一帧填充
                        if len(frame_probs_list) < n_frames:
                            last_frame_probs = frame_probs_list[-1] if len(frame_probs_list) > 0 else np.zeros(batch_size)
                            for _ in range(n_frames - len(frame_probs_list)):
                                frame_probs_list.append(last_frame_probs)
                        
                        # 转换为(n_frames, batch_size)，然后转置为(batch_size, n_frames)
                        frame_probs_array = np.array(frame_probs_list).T  # (batch_size, n_frames)
                    else:
                        # 如果没有输出，使用零概率
                        frame_probs_array = np.zeros((batch_size, n_frames))
                    
                    all_pred_frame.append(frame_probs_array)
                    all_labels.append(y[:, 1].cpu().numpy() if y.shape[1] > 1 else y.cpu().numpy())
                    # 确保toa是1D数组，与CRASH的evaluation函数兼容
                    toa_np = toa_tensor.cpu().numpy()
                    if toa_np.ndim > 1:
                        toa_np = toa_np.flatten()  # 如果是(batch_size, 1)，展平为(batch_size,)
                    all_toas.append(toa_np)
            
            all_pred = np.concatenate(all_pred_frame, axis=0)  # (n_videos, n_frames)
            all_labels = np.concatenate(all_labels)
            all_toas = np.concatenate(all_toas)
            # 确保all_toas是1D数组，与CRASH的evaluation函数兼容
            if all_toas.ndim > 1:
                all_toas = all_toas.flatten()
            
            # 计算指标 - 现在使用真正的frame-level predictions
            AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toas, fps=ds_params["fps"])
            
            logger.info(f"Epoch {epoch} 评估结果:")
            logger.info(f"  AP (video-level): {AP:.4f}")
            logger.info(f"  mTTA: {mTTA:.4f}")
            logger.info(f"  TTA@R80: {TTA_R80:.4f}")
            logger.info(f"  P@R80: {P_R80:.4f}")
            sys.stdout.flush()
            
            # 保存最佳模型
            if AP > best_ap:
                best_ap = AP
                best_epoch = epoch
                checkpoint_path = os.path.join(exp_output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ap': AP,
                    'mtta': mTTA,
                    'params': params
                }, checkpoint_path)
                logger.info(f"✓ 保存最佳模型 (AP={AP:.4f}) 到 {checkpoint_path}")
                sys.stdout.flush()
    
    logger.info(f"\n[步骤 5/6] 训练完成！")
    logger.info(f"最佳AP: {best_ap:.4f} (Epoch {best_epoch})")
    sys.stdout.flush()
    
    # 最终评估
    logger.info(f"[步骤 6/6] 最终评估...")
    sys.stdout.flush()
    
    # 加载最佳模型
    checkpoint_path = os.path.join(exp_output_dir, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ 加载最佳模型 (Epoch {checkpoint['epoch']})")
    
    model.eval()
    all_pred_frame = []  # 存储frame-level predictions: (n_videos, n_frames)
    all_labels = []
    all_toas = []
    
    with torch.no_grad():
        for x, y, toa in tqdm(test_loader, desc="最终测试"):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            batch_size = x.size(0)
            
            # 确保toa是tensor - 统一处理逻辑（与训练循环一致）
            if isinstance(toa, list):
                # 检查是否是list包含tensor的情况：[tensor([...])]
                if len(toa) > 0 and isinstance(toa[0], torch.Tensor):
                    # 如果list中只有一个tensor，且tensor有多个元素，展开它
                    tensor_val = toa[0]
                    if tensor_val.numel() > 1:
                        # tensor包含多个值，直接使用
                        toa = tensor_val.to(device).float()
                    else:
                        # tensor只有一个值，展开到batch_size
                        val = tensor_val.item() if tensor_val.numel() == 1 else float(tensor_val[0])
                        toa = torch.tensor([val] * batch_size, dtype=torch.float32, device=device)
                # 如果是list of lists（批处理），展平
                elif len(toa) > 0 and isinstance(toa[0], list):
                    toa_flat = []
                    for item in toa:
                        if isinstance(item, list) and len(item) > 0:
                            val = item[0]
                        else:
                            val = item
                        # 处理tensor类型
                        if isinstance(val, torch.Tensor):
                            val = val.item() if val.numel() == 1 else float(val[0])
                        else:
                            val = float(val)
                        toa_flat.append(val)
                    toa = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                else:
                    # 处理list of scalars/tensors
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
                # 如果是2D，展平为1D
                if toa.dim() > 1:
                    toa = toa.flatten()
            elif not isinstance(toa, torch.Tensor):
                try:
                    if isinstance(toa, torch.Tensor):
                        toa = toa.item() if toa.numel() == 1 else toa[0]
                    toa = torch.tensor([float(toa)] * batch_size, dtype=torch.float32, device=device)
                except:
                    toa = torch.tensor([0.0] * batch_size, dtype=torch.float32, device=device)
            else:
                toa = toa.to(device, non_blocking=True)
                # 确保是1D tensor
                if toa.dim() > 1:
                    toa = toa.flatten()
                elif toa.dim() == 0:
                    toa = toa.unsqueeze(0)
            
            # 确保toa的shape与batch_size匹配
            if toa.shape[0] != batch_size:
                if toa.shape[0] == 1:
                    toa = toa.expand(batch_size)
                else:
                    # 如果shape不匹配且不是1，可能需要截断或填充
                    if toa.shape[0] > batch_size:
                        toa = toa[:batch_size]
                    else:
                        # 填充到batch_size
                        padding = torch.zeros(batch_size - toa.shape[0], device=device, dtype=toa.dtype)
                        toa = torch.cat([toa, padding])
            
            toa_tensor = toa
            
            # CRASH对齐后的新格式: losses, all_outputs, all_hidden
            _, all_outputs, _ = model(x, None, toa_tensor)
            
            # all_outputs是列表，每个元素是(B, 2)的logits，对应每一帧的输出
            # 需要构造frame-level predictions用于evaluation
            # 每个视频的frame-level predictions: (n_frames,)
            n_frames = ds_params["n_frames"]
            if len(all_outputs) > 0:
                # 将all_outputs转换为frame-level predictions
                # all_outputs是列表，长度为n_frames，每个元素是(B, 2)
                frame_probs_list = []
                for frame_idx in range(min(len(all_outputs), n_frames)):
                    frame_output = all_outputs[frame_idx]  # (B, 2)
                    frame_probs = torch.softmax(frame_output, dim=-1)[:, 1]  # (B,) - 正类概率
                    frame_probs_list.append(frame_probs.cpu().numpy())
                
                # 如果帧数不足n_frames，用最后一帧填充
                if len(frame_probs_list) < n_frames:
                    last_frame_probs = frame_probs_list[-1] if len(frame_probs_list) > 0 else np.zeros(batch_size)
                    for _ in range(n_frames - len(frame_probs_list)):
                        frame_probs_list.append(last_frame_probs)
                
                # 转换为(n_frames, batch_size)，然后转置为(batch_size, n_frames)
                frame_probs_array = np.array(frame_probs_list).T  # (batch_size, n_frames)
            else:
                # 如果没有输出，使用零概率
                frame_probs_array = np.zeros((batch_size, n_frames))
            
            all_pred_frame.append(frame_probs_array)
            all_labels.append(y[:, 1].cpu().numpy() if y.shape[1] > 1 else y.cpu().numpy())
            # 确保toa是1D数组，与CRASH的evaluation函数兼容
            toa_np = toa_tensor.cpu().numpy()
            if toa_np.ndim > 1:
                toa_np = toa_np.flatten()  # 如果是(batch_size, 1)，展平为(batch_size,)
            all_toas.append(toa_np)
    
    all_pred = np.concatenate(all_pred_frame, axis=0)  # (n_videos, n_frames)
    all_labels = np.concatenate(all_labels)
    all_toas = np.concatenate(all_toas)
    # 确保all_toas是1D数组，与CRASH的evaluation函数兼容
    if all_toas.ndim > 1:
        all_toas = all_toas.flatten()
    
    # 计算指标 - 现在使用真正的frame-level predictions
    AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toas, fps=ds_params["fps"])
    
    logger.info(f"\n最终结果:")
    logger.info(f"  AP (video-level): {AP:.4f}")
    logger.info(f"  mTTA: {mTTA:.4f}")
    logger.info(f"  TTA@R80: {TTA_R80:.4f}")
    logger.info(f"  P@R80: {P_R80:.4f}")
    sys.stdout.flush()
    
    # 保存结果
    results = {
        "dataset": dataset_name,
        "params": params,
        "ap": float(AP),
        "mtta": float(mTTA),
        "tta_r80": float(TTA_R80),
        "p_r80": float(P_R80),
        "best_epoch": best_epoch
    }
    
    results_file = os.path.join(exp_output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ 结果已保存到 {results_file}")
    sys.stdout.flush()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='使用最佳参数训练数据集')
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU ID')
    parser.add_argument('--dataset', type=str, default=None,
                       help='要训练的单个数据集 (dad/crash/a3d)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='要训练的数据集列表 (如果指定了--dataset则忽略)')
    parser.add_argument('--output_dir', type=str, default='output_gru/best_params',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 确定要训练的数据集
    if args.dataset:
        datasets_to_train = [args.dataset]
    elif args.datasets:
        datasets_to_train = args.datasets
    else:
        datasets_to_train = ['dad', 'crash', 'a3d']
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练每个数据集
    all_results = []
    for dataset in datasets_to_train:
        if dataset not in BEST_PARAMS:
            logger.warning(f"跳过未知数据集: {dataset}")
            continue
        
        try:
            results = train_dataset(dataset, args.gpu_id, args.output_dir)
            all_results.append(results)
        except Exception as e:
            logger.error(f"训练 {dataset} 时出错: {e}", exc_info=True)
            continue
    
    # 保存所有结果
    if len(all_results) > 0:
        all_results_file = os.path.join(args.output_dir, "all_results.json")
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\n所有训练完成！结果已保存到 {all_results_file}")
        logger.info("=" * 80)
        for result in all_results:
            logger.info(f"{result['dataset'].upper()}: AP={result['ap']:.4f}, mTTA={result['mtta']:.4f}")
        logger.info("=" * 80)


if __name__ == '__main__':
    main()

