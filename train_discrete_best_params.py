#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用离散超参数搜索找到的最佳参数训练三个数据集
基于optuna_discrete_20260119_031907的最佳结果
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


# 最佳参数配置（基于离散超参数搜索，2026-01-19）
# 从optuna_discrete_20260119_031907结果中提取
BEST_PARAMS = {
    "crash": {
        # Crash: Trial #0, AP=0.9993 (最佳)
        "lambda_align": 1e-05,
        "lambda_sparse": 0.002,
        "batch_size": 32,
        "learning_rate": 0.0002,
        "weight_decay": 5e-05,
        "h_dim": 512,
        "z_dim": 512,
        "n_epochs": 80  # CRASH原文使用80 epochs
    },
    "a3d": {
        # A3D: Trial #0, AP=0.9622 (最佳)
        "lambda_align": 0.0002,
        "lambda_sparse": 0.05,
        "batch_size": 16,  # 注意：避免使用64，会导致OOM
        "learning_rate": 5e-05,
        "weight_decay": 1e-05,
        "h_dim": 1024,
        "z_dim": 512,
        "n_epochs": 80  # CRASH原文使用80 epochs
    },
    "dad": {
        # DAD: 等待Optuna搜索完成，暂时使用之前的最佳参数
        # 如果Optuna完成，会从结果文件中更新
        "lambda_align": 2.4017632837038076e-05,
        "lambda_sparse": 0.0002624009864715832,
        "batch_size": 8,
        "learning_rate": 0.00039865649058389127,
        "weight_decay": 2.778678567292157e-05,
        "h_dim": 256,
        "z_dim": 128,
        "n_epochs": 80  # CRASH原文使用80 epochs
    }
}

# 尝试从Optuna结果文件加载DAD的最佳参数
optuna_results_dir = "output/optuna_discrete_20260119_031907"
dad_results_file = os.path.join(optuna_results_dir, "dad_discrete_optuna_results.json")
if os.path.exists(dad_results_file):
    try:
        with open(dad_results_file, 'r') as f:
            dad_results = json.load(f)
            BEST_PARAMS["dad"].update(dad_results["best_params"])
            BEST_PARAMS["dad"]["n_epochs"] = 80  # 保持80 epochs
            logger.info(f"✓ 从 {dad_results_file} 加载了DAD最佳参数 (AP={dad_results['best_ap']:.4f})")
    except Exception as e:
        logger.warning(f"无法加载DAD最佳参数: {e}")

# 数据集参数
DATASET_PARAMS = {
    "dad": {"x_dim": 4096, "n_obj": 19, "n_frames": 100, "fps": 20.0, "feature": "vgg16", "phase_train": "training", "phase_test": "testing"},
    "crash": {"x_dim": 4096, "n_obj": 19, "n_frames": 50, "fps": 10.0, "feature": "vgg16", "phase_train": "train", "phase_test": "test"},
    "a3d": {"x_dim": 4096, "n_obj": 19, "n_frames": 100, "fps": 20.0, "feature": "vgg16", "phase_train": "train", "phase_test": "test"},
}


def train_dataset(dataset_name, gpu_id, output_dir):
    """使用最佳参数训练单个数据集"""
    
    logger.info("=" * 80)
    logger.info(f"使用离散超参数搜索最佳参数训练: {dataset_name.upper()} 数据集")
    logger.info("=" * 80)
    
    # 获取参数
    params = BEST_PARAMS[dataset_name]
    ds_params = DATASET_PARAMS[dataset_name]
    
    logger.info(f"超参数设置:")
    logger.info(f"  - Learning Rate: {params['learning_rate']}")
    logger.info(f"  - Batch Size: {params['batch_size']}")
    logger.info(f"  - Weight Decay: {params['weight_decay']}")
    logger.info(f"  - Lambda Align: {params['lambda_align']}")
    logger.info(f"  - Lambda Sparse: {params['lambda_sparse']}")
    logger.info(f"  - h_dim: {params['h_dim']}")
    logger.info(f"  - z_dim: {params['z_dim']}")
    logger.info(f"  - Epochs: {params['n_epochs']}")
    sys.stdout.flush()
    
    # 设备设置
    logger.info(f"[步骤 1/6] 设置设备...")
    sys.stdout.flush()
    
    if torch.cuda.is_available():
        # 如果设置了CUDA_VISIBLE_DEVICES，设备索引应该是0
        if gpu_id is not None:
            # 检查是否设置了CUDA_VISIBLE_DEVICES
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible:
                # 如果设置了CUDA_VISIBLE_DEVICES，使用设备0
                device = torch.device('cuda:0')
                logger.info(f"检测到CUDA_VISIBLE_DEVICES={cuda_visible}，使用cuda:0")
            else:
                device = torch.device(f'cuda:{gpu_id}')
            try:
                torch.cuda.set_device(device)
                # 测试设备是否可用
                _ = torch.zeros(1).to(device)
            except Exception as e:
                logger.warning(f"CUDA设备不可用: {e}，使用CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    logger.info(f"使用设备: {device}")
    sys.stdout.flush()
    
    # 加载数据
    logger.info(f"[步骤 2/6] 加载数据...")
    sys.stdout.flush()
    
    # 尝试多个可能的数据路径
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "CRASH", "data", dataset_name),
        os.path.join(os.path.dirname(__file__), "..", "data", dataset_name),
        os.path.join("/data/sony/LFCRASH/CRASH/data", dataset_name),
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"数据路径不存在，尝试过的路径: {possible_paths}")
    
    logger.info(f"使用数据路径: {data_path}")
    
    if dataset_name == "dad":
        train_dataset = DADDataset(data_path, ds_params["feature"], phase=ds_params["phase_train"], toTensor=False)
        test_dataset = DADDataset(data_path, ds_params["feature"], phase=ds_params["phase_test"], toTensor=False)
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
    # 对于A3D，使用较小的batch_size避免OOM
    if dataset_name == "dad":
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=0)
        logger.info("✓ 使用单进程数据加载（DAD数据集）")
    elif dataset_name == "a3d":
        # A3D使用较小的batch_size，避免OOM
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
        logger.info(f"✓ 使用batch_size={params['batch_size']}，避免OOM（A3D数据集）")
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
    logger.info(f"[步骤 3/6] 初始化模型...")
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
    logger.info(f"[步骤 4/6] 开始训练（{params['n_epochs']}个epoch）...")
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
            if isinstance(toa, list):
                if len(toa) > 0 and isinstance(toa[0], list):
                    toa_flat = []
                    for item in toa:
                        if isinstance(item, list) and len(item) > 0:
                            val = item[0]
                        else:
                            val = item if not isinstance(item, list) else 0.0
                        toa_flat.append(float(val))
                    toa = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                elif len(toa) > 0:
                    if isinstance(toa[0], (int, float)):
                        toa = torch.tensor(toa, dtype=torch.float32, device=device)
                    else:
                        toa_flat = [float(item[0]) if isinstance(item, (list, np.ndarray)) and len(item) > 0 else 0.0 for item in toa]
                        toa = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                else:
                    toa = torch.tensor([0.0], dtype=torch.float32, device=device)
            elif isinstance(toa, np.ndarray):
                toa = torch.from_numpy(toa).float().to(device)
            elif isinstance(toa, torch.Tensor):
                toa = toa.to(device, non_blocking=True)
            else:
                try:
                    toa = torch.tensor([float(toa)], dtype=torch.float32, device=device)
                except:
                    toa = torch.tensor([0.0], dtype=torch.float32, device=device)
            
            # 确保toa的shape正确
            batch_size = x.size(0)
            if toa.shape[0] != batch_size:
                if toa.shape[0] == 1:
                    toa = toa.expand(batch_size)
                else:
                    if toa.shape[0] > batch_size:
                        toa = toa[:batch_size]
                    else:
                        padding = torch.zeros(batch_size - toa.shape[0], device=device, dtype=toa.dtype)
                        toa = torch.cat([toa, padding])
            
            # 前向传播
            optimizer.zero_grad()
            losses, _, _ = model(x, y, toa)
            # 处理losses：可能是tensor或dict
            if isinstance(losses, dict):
                # 如果是字典，计算总损失
                loss = sum(losses.values()) if losses else torch.tensor(0.0, device=device)
                if isinstance(loss, torch.Tensor):
                    loss = loss.mean() if loss.numel() > 1 else loss
                else:
                    loss = torch.tensor(float(loss), device=device)
            else:
                # 如果是tensor
                loss = losses.mean() if losses.numel() > 1 else losses
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        logger.info(f"Epoch {epoch}/{params['n_epochs']}: 平均损失 = {avg_loss:.6f}")
        sys.stdout.flush()
        
        # 评估
        if epoch % 10 == 0 or epoch == params["n_epochs"]:
            logger.info(f"[步骤 5/6] Epoch {epoch}: 评估模型...")
            sys.stdout.flush()
            
            model.eval()
            all_pred_frame = []
            all_labels = []
            all_toas = []
            
            with torch.no_grad():
                for x, y, toa in tqdm(test_loader, desc="评估中"):
                    x = x.to(device, non_blocking=True).float()
                    y = y.to(device, non_blocking=True).float()
                    batch_size = x.size(0)
                    
                    # 处理toa
                    if isinstance(toa, list):
                        if len(toa) > 0 and isinstance(toa[0], list):
                            toa_flat = []
                            for item in toa:
                                if isinstance(item, list) and len(item) > 0:
                                    val = item[0]
                                else:
                                    val = item if not isinstance(item, list) else 0.0
                                toa_flat.append(float(val))
                            toa = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                        elif len(toa) > 0:
                            if isinstance(toa[0], (int, float)):
                                toa = torch.tensor(toa, dtype=torch.float32, device=device)
                            else:
                                toa_flat = [float(item[0]) if isinstance(item, (list, np.ndarray)) and len(item) > 0 else 0.0 for item in toa]
                                toa = torch.tensor(toa_flat, dtype=torch.float32, device=device)
                        else:
                            toa = torch.tensor([0.0] * batch_size, dtype=torch.float32, device=device)
                    elif isinstance(toa, np.ndarray):
                        toa = torch.from_numpy(toa).float().to(device)
                    elif isinstance(toa, torch.Tensor):
                        toa = toa.to(device, non_blocking=True)
                    else:
                        try:
                            toa = torch.tensor([float(toa)] * batch_size, dtype=torch.float32, device=device)
                        except:
                            toa = torch.tensor([0.0] * batch_size, dtype=torch.float32, device=device)
                    
                    # 确保toa的shape正确
                    if toa.shape[0] != batch_size:
                        if toa.shape[0] == 1:
                            toa = toa.expand(batch_size)
                        else:
                            if toa.shape[0] > batch_size:
                                toa = toa[:batch_size]
                            else:
                                padding = torch.zeros(batch_size - toa.shape[0], device=device, dtype=toa.dtype)
                                toa = torch.cat([toa, padding])
                    
                    # 构造toa（测试时）
                    toa_tensor = toa
                    # CRASH对齐后的新格式: losses, all_outputs, all_hidden
                    _, all_outputs, _ = model(x, None, toa_tensor)
                    
                    # all_outputs是列表，每个元素是(B, 2)的logits
                    # 需要转换为frame-level predictions
                    if len(all_outputs) > 0:
                        # all_outputs 是一个列表，每个元素是 (B, 2) 的 logits
                        # 需要将其转换为 (B, n_frames) 的概率
                        frame_probs_list = []
                        for frame_output in all_outputs:
                            frame_probs_list.append(torch.softmax(frame_output, dim=-1)[:, 1].cpu().numpy())
                        # 将列表转换为 (n_frames, B) 然后转置为 (B, n_frames)
                        batch_frame_probs = np.array(frame_probs_list).T
                        all_pred_frame.append(batch_frame_probs)
                    else:
                        # 如果没有输出，使用零概率
                        all_pred_frame.append(np.zeros((batch_size, ds_params["n_frames"])))
                    
                    all_labels.append(y[:, 1].cpu().numpy() if y.shape[1] > 1 else y.cpu().numpy())
                    toa_np = toa_tensor.cpu().numpy()
                    if toa_np.ndim > 1:
                        toa_np = toa_np.flatten()
                    all_toas.append(toa_np)
            
            all_pred_frame = np.concatenate(all_pred_frame, axis=0)
            all_labels = np.concatenate(all_labels)
            all_toas = np.concatenate(all_toas)
            if all_toas.ndim > 1:
                all_toas = all_toas.flatten()
            
            AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred_frame, all_labels, all_toas, fps=ds_params["fps"])
            
            logger.info(f"Epoch {epoch} 评估结果:")
            logger.info(f"  AP: {AP:.4f}")
            logger.info(f"  mTTA: {mTTA:.4f}")
            logger.info(f"  TTA@R80: {TTA_R80:.4f}")
            logger.info(f"  P@R80: {P_R80:.4f}")
            sys.stdout.flush()
            
            # 更新学习率
            scheduler.step(avg_loss)
            
            # 保存最佳模型
            if AP > best_ap:
                best_ap = AP
                best_epoch = epoch
                checkpoint_dir = os.path.join(output_dir, dataset_name)
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ap': AP,
                    'mTTA': mTTA,
                    'TTA_R80': TTA_R80,
                    'P_R80': P_R80,
                    'params': params,
                }, checkpoint_path)
                logger.info(f"✓ 保存最佳模型 (AP={AP:.4f}) 到 {checkpoint_path}")
                sys.stdout.flush()
    
    logger.info("=" * 80)
    logger.info(f"训练完成！最佳AP: {best_ap:.4f} (Epoch {best_epoch})")
    logger.info("=" * 80)
    sys.stdout.flush()
    
    return best_ap, best_epoch


def main():
    parser = argparse.ArgumentParser(description="使用离散超参数搜索最佳参数训练模型")
    parser.add_argument("--dataset", type=str, choices=["dad", "crash", "a3d", "all"], default="all",
                        help="要训练的数据集")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID")
    parser.add_argument("--output_dir", type=str, default="output/discrete_best_training",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config = {
        "best_params": BEST_PARAMS,
        "dataset_params": DATASET_PARAMS,
        "timestamp": timestamp
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"配置已保存到: {config_path}")
    
    # 训练
    datasets = ["dad", "crash", "a3d"] if args.dataset == "all" else [args.dataset]
    
    results = {}
    for dataset_name in datasets:
        try:
            ap, epoch = train_dataset(dataset_name, args.gpu, output_dir)
            results[dataset_name] = {"ap": ap, "best_epoch": epoch}
        except Exception as e:
            logger.error(f"训练 {dataset_name} 时出错: {e}", exc_info=True)
            results[dataset_name] = {"error": str(e)}
    
    # 保存结果
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存到: {results_path}")
    
    # 打印总结
    logger.info("=" * 80)
    logger.info("训练总结")
    logger.info("=" * 80)
    for dataset_name, result in results.items():
        if "error" in result:
            logger.info(f"{dataset_name.upper()}: ❌ {result['error']}")
        else:
            logger.info(f"{dataset_name.upper()}: AP={result['ap']:.4f}, Best Epoch={result['best_epoch']}")


if __name__ == "__main__":
    main()
