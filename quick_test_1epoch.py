#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本：只训练1个epoch并测试可视化
用于验证全流程是否正常工作
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
import matplotlib.pyplot as plt

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


def visualize_predictions(all_pred, all_labels, all_toas, output_dir, dataset_name, fps=20.0):
    """
    可视化预测结果
    参考CRASH的visualization功能
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择前5个样本进行可视化
    n_samples = min(5, len(all_pred))
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for idx in range(n_samples):
        pred = all_pred[idx]
        label = all_labels[idx]
        toa = all_toas[idx]
        
        # 转换为numpy数组
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        
        # 获取时间轴（秒）
        n_frames = len(pred)
        time_axis = np.arange(n_frames) / fps
        
        # 绘制预测曲线
        axes[idx].plot(time_axis, pred, 'b-', label='Prediction', linewidth=2, alpha=0.7)
        
        # 绘制真实标签
        if label > 0:
            # 事故发生在toa帧
            accident_frame = int(toa)
            axes[idx].axvline(x=accident_frame/fps, color='r', linestyle='--', 
                            label=f'Accident @ {accident_frame/fps:.1f}s', linewidth=2)
            axes[idx].axhline(y=1.0, color='g', linestyle=':', alpha=0.5, label='Threshold')
        
        axes[idx].set_xlabel('Time (seconds)', fontsize=10)
        axes[idx].set_ylabel('Prediction Score', fontsize=10)
        axes[idx].set_title(f'Sample {idx+1} - {dataset_name}', fontsize=12)
        axes[idx].legend(loc='best', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([-0.1, 1.1])
    
    plt.tight_layout()
    vis_path = os.path.join(output_dir, f'{dataset_name}_visualization.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ 可视化结果已保存到: {vis_path}")
    return vis_path


def train_dataset_quick_test(dataset_name, gpu_id=0, output_dir="output/quick_test"):
    """
    快速测试：只训练1个epoch并测试可视化
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"快速测试数据集: {dataset_name.upper()}")
    logger.info(f"{'='*80}\n")
    
    # 数据集配置 - 参考train_best_params.py
    DATASET_CONFIGS = {
        "dad": {
            "dataset_class": DADDataset,
            "data_path": "/data/sony/LFCRASH/CRASH/data/dad",
            "feature": "vgg16",
            "x_dim": 4096,
            "n_obj": 19,
            "n_frames": 100,
            "fps": 20.0,
            "h_dim": 256,
            "z_dim": 128,
            "lambda_align": 2.4017632837038076e-05,
            "lambda_sparse": 0.0002624009864715832,
            "batch_size": 8,
            "learning_rate": 0.00039865649058389127,
            "weight_decay": 2.778678567292157e-05,
        },
        "crash": {
            "dataset_class": CrashDataset,
            "data_path": "/data/sony/LFCRASH/CRASH/data/crash",
            "feature": "vgg16",
            "x_dim": 4096,
            "n_obj": 19,
            "n_frames": 50,
            "fps": 10.0,
            "h_dim": 512,
            "z_dim": 256,
            "lambda_align": 0.0008750829573052726,
            "lambda_sparse": 0.022156161895046517,
            "batch_size": 16,
            "learning_rate": 0.00023974291760359517,
            "weight_decay": 2.3725279765274536e-05,
        },
        "a3d": {
            "dataset_class": A3DDataset,
            "data_path": "/data/sony/LFCRASH/CRASH/data/a3d",
            "feature": "vgg16",
            "x_dim": 4096,
            "n_obj": 19,
            "n_frames": 100,
            "fps": 20.0,
            "h_dim": 768,
            "z_dim": 256,
            "lambda_align": 0.0001381887119104739,
            "lambda_sparse": 0.009361841980131274,
            "batch_size": 32,
            "learning_rate": 1.0421360243636988e-05,
            "weight_decay": 3.0512968535137967e-05,
        }
    }
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    ds_params = DATASET_CONFIGS[dataset_name]
    
    # 设备配置
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 加载数据集
    logger.info(f"[步骤 1/5] 加载数据集: {ds_params['data_path']}")
    phase_map = {
        "dad": {"train": "training", "test": "testing"},
        "crash": {"train": "train", "test": "test"},
        "a3d": {"train": "train", "test": "test"}
    }
    train_phase = phase_map[dataset_name]["train"]
    test_phase = phase_map[dataset_name]["test"]
    
    train_dataset = ds_params["dataset_class"](
        ds_params["data_path"], 
        feature=ds_params["feature"],
        phase=train_phase
    )
    test_dataset = ds_params["dataset_class"](
        ds_params["data_path"], 
        feature=ds_params["feature"],
        phase=test_phase
    )
    
    train_loader = DataLoader(train_dataset, batch_size=ds_params["batch_size"], 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=ds_params["batch_size"], 
                             shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  测试集: {len(test_dataset)} 样本")
    
    # 初始化模型
    logger.info(f"[步骤 2/5] 初始化模型...")
    
    # 加载概念文件
    concept_file = os.path.join(os.path.dirname(__file__), "..", "000_all_concept_set.txt")
    if not os.path.exists(concept_file):
        concept_file = None
        logger.warning("未找到概念文件，将不使用概念层")
    
    model = LFCRASH_CBM_GRU(
        x_dim=ds_params["x_dim"],
        h_dim=ds_params["h_dim"],
        z_dim=ds_params["z_dim"],
        n_layers=2,
        n_obj=ds_params["n_obj"],
        n_frames=ds_params["n_frames"],
        fps=ds_params["fps"],
        with_saa=True,
        num_concepts=837,
        concept_file=concept_file,
        lambda_align=ds_params["lambda_align"],
        lambda_sparse=ds_params["lambda_sparse"],
        device=device
    ).to(device)
    
    logger.info("✓ 模型初始化完成")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=ds_params["learning_rate"], 
                          weight_decay=ds_params["weight_decay"])
    
    # 训练1个epoch
    logger.info(f"[步骤 3/5] 开始训练（1个epoch）...")
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc="Epoch [1/1]")
    for batch_idx, (x, y, toa) in enumerate(pbar):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        
        # 处理toa - 参考train_best_params.py的处理方式
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
        elif isinstance(toa, torch.Tensor):
            # 如果已经是tensor，确保在正确的设备上
            toa = toa.to(device)
        else:
            toa = torch.tensor([float(toa)], dtype=torch.float32, device=device)
        
        optimizer.zero_grad()
        
        # 前向传播 - 返回 losses, all_outputs, all_hidden
        losses, all_outputs, all_hidden = model(x, y, toa)
        
        # 获取总损失
        total_loss = losses.get("total_loss", losses.get("total", 0.0))
        if isinstance(total_loss, (int, float)) and total_loss == 0.0:
            total_loss = losses.get("cross_entropy", 0.0)
            if "auxloss" in losses:
                total_loss = total_loss + losses["auxloss"]
        
        # 获取各个损失用于显示
        ce_loss = losses.get("cross_entropy", total_loss)
        aux_loss = losses.get("auxloss", torch.tensor(0.0))
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        n_batches += 1
        
        ce_val = ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss
        aux_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'ce': f'{ce_val:.4f}',
            'aux': f'{aux_val:.4f}'
        })
    
    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
    logger.info(f"\nEpoch 1 完成: Total={avg_loss:.4f}")
    
    # 评估
    logger.info(f"[步骤 4/5] 评估模型...")
    model.eval()
    all_pred = []
    all_labels = []
    all_toas = []
    
    with torch.no_grad():
        for x, y, toa in tqdm(test_loader, desc="评估中"):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            
            # 处理toa - 与训练时相同的处理方式
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
            elif isinstance(toa, torch.Tensor):
                toa = toa.to(device)
            else:
                toa = torch.tensor([float(toa)], dtype=torch.float32, device=device)
            
            losses, all_outputs, all_hidden = model(x, None, toa)
            
            # all_outputs是列表，每个元素是(B, 2)的logits
            # 需要转换为frame-level预测 (B, T)
            batch_size = x.size(0)
            n_frames = ds_params["n_frames"]
            
            if isinstance(all_outputs, list) and len(all_outputs) > 0:
                # 将列表转换为tensor: (T, B, 2) -> (B, T, 2)
                outputs_tensor = torch.stack(all_outputs, dim=1)  # (B, T, 2)
                # 取positive类的概率: (B, T, 2) -> (B, T)
                pred_probs = torch.softmax(outputs_tensor, dim=-1)[:, :, 1]  # (B, T)
            else:
                # 如果没有输出，使用零概率
                pred_probs = torch.zeros(batch_size, n_frames, device=device)
            
            all_pred.append(pred_probs.cpu())
            all_labels.append(y.cpu())
            all_toas.append(toa.cpu())
    
    # 合并所有批次
    # all_pred是(B, T)形状，需要拼接
    all_pred = torch.cat(all_pred, dim=0).numpy()  # (N, T)
    all_labels_raw = torch.cat(all_labels, dim=0).numpy()
    all_toas = torch.cat(all_toas, dim=0).numpy()
    
    # 处理labels：如果是frame-level数组，转换为video-level标量
    # DAD的labels可能是(10, 2)的frame-level数组，需要提取video-level标签
    all_labels = []
    for idx, label in enumerate(all_labels_raw):
        if isinstance(label, np.ndarray) and len(label.shape) > 0:
            # 如果是frame-level数组，检查是否有positive帧
            if len(label.shape) == 2:
                # (n_frames, 2)格式：检查是否有任何帧是positive
                video_label = float(np.any(label[:, 1] > 0) if label.shape[1] > 1 else np.any(label > 0))
            elif len(label.shape) == 1 and len(label) > 1:
                # (2,)格式：video-level标签，取第二个元素
                video_label = float(label[1] > 0)
            else:
                # 标量数组，取第一个元素
                video_label = float(label[0] > 0) if len(label) > 0 else 0.0
        else:
            # 已经是标量
            video_label = float(label > 0) if isinstance(label, (int, float, np.number)) else 0.0
        all_labels.append(video_label)
    all_labels = np.array(all_labels)
    
    # 计算指标
    logger.info("计算评估指标...")
    try:
        AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, all_toas, fps=ds_params["fps"])
        logger.info(f"\n评估结果:")
        logger.info(f"  AP: {AP:.4f}")
        logger.info(f"  mTTA: {mTTA:.4f}")
        logger.info(f"  TTA@R80: {TTA_R80:.4f}")
        logger.info(f"  P@R80: {P_R80:.4f}")
    except Exception as e:
        logger.error(f"评估时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 可视化
    logger.info(f"[步骤 5/5] 生成可视化...")
    try:
        vis_path = visualize_predictions(all_pred, all_labels, all_toas, 
                                        dataset_output_dir, dataset_name, 
                                        fps=ds_params["fps"])
        logger.info(f"✓ 可视化完成")
    except Exception as e:
        logger.error(f"可视化时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存结果
    results = {
        "dataset": dataset_name,
        "epoch": 1,
        "loss": avg_loss,
        "AP": float(AP),
        "mTTA": float(mTTA),
        "TTA@R80": float(TTA_R80),
        "P@R80": float(P_R80),
        "visualization": vis_path if 'vis_path' in locals() else None
    }
    
    results_path = os.path.join(dataset_output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ 结果已保存到: {results_path}")
    logger.info(f"\n{'='*80}")
    logger.info(f"快速测试完成！")
    logger.info(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='快速测试：1个epoch + 可视化')
    parser.add_argument('--dataset', type=str, choices=['dad', 'crash', 'a3d'], 
                       default='dad', help='数据集名称')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--output_dir', type=str, default='output/quick_test',
                       help='输出目录')
    
    args = parser.parse_args()
    
    try:
        results = train_dataset_quick_test(args.dataset, args.gpu_id, args.output_dir)
        if results:
            logger.info("✅ 快速测试成功完成！")
        else:
            logger.error("❌ 快速测试失败")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 训练 {args.dataset} 时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
