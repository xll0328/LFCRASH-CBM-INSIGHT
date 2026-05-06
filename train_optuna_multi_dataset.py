#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据集Optuna超参数搜索
支持为不同数据集设计不同的搜索空间
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
import optuna
from optuna.trial import TrialState

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
# 添加CRASH目录到路径
crash_path = os.path.join(os.path.dirname(__file__), "..", "CRASH")
if os.path.exists(crash_path):
    sys.path.insert(0, crash_path)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入eval_tools（从CRASH目录）
try:
    from src.eval_tools import evaluation
    eval_tools_imported = True
    # 创建compute_ap包装函数
    def compute_ap(all_labels, all_probs, fps=20.0, n_frames=100):
        """计算AP的包装函数"""
        # 将probs转换为frame-level predictions
        # all_probs是video-level的，需要转换为frame-level
        if len(all_probs.shape) == 1:
            # 如果是1D，假设是video-level，需要扩展
            # 这里简化处理，直接使用evaluation函数
            # 需要构造all_pred和time_of_accidents
            n_videos = len(all_probs)
            all_pred = np.tile(all_probs[:, None], (1, n_frames))
            time_of_accidents = np.where(all_labels > 0, n_frames, n_frames + 1)
        else:
            all_pred = all_probs
            n_frames = all_pred.shape[1] if len(all_pred.shape) > 1 else 100
            time_of_accidents = np.where(all_labels > 0, n_frames, n_frames + 1)
        
        # 确保数组长度一致
        min_len = min(len(all_pred), len(all_labels), len(time_of_accidents))
        if min_len == 0:
            return 0.0  # 如果没有数据，返回0
        
        # 确保索引不会越界（evaluation函数内部可能使用len-1作为索引）
        if min_len < len(all_pred):
            all_pred = all_pred[:min_len]
        if min_len < len(all_labels):
            all_labels = all_labels[:min_len]
        if min_len < len(time_of_accidents):
            time_of_accidents = time_of_accidents[:min_len]
        
        # 确保time_of_accidents是1D数组
        if time_of_accidents.ndim > 1:
            time_of_accidents = time_of_accidents.flatten()
        
        try:
            AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, time_of_accidents, fps=fps)
            return AP
        except (IndexError, ValueError) as e:
            # 如果evaluation函数出错，返回0（表示性能不佳）
            return 0.0
    
    def compute_tta(all_labels, all_probs, fps=20.0):
        """计算TTA的包装函数"""
        if len(all_probs.shape) == 1:
            n_videos = len(all_probs)
            n_frames = 100
            all_pred = np.tile(all_probs[:, None], (1, n_frames))
            time_of_accidents = np.where(all_labels > 0, n_frames, n_frames + 1)
        else:
            all_pred = all_probs
            time_of_accidents = np.where(all_labels > 0, all_pred.shape[1], all_pred.shape[1] + 1)
        
        AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, time_of_accidents, fps=fps)
        return mTTA
except ImportError:
    try:
        sys.path.insert(0, crash_path)
        from src.eval_tools import evaluation
        eval_tools_imported = True
        # 创建包装函数（同上）
        def compute_ap(all_labels, all_probs, fps=20.0, n_frames=100):
            if len(all_probs.shape) == 1:
                n_videos = len(all_probs)
                all_pred = np.tile(all_probs[:, None], (1, n_frames))
                time_of_accidents = np.where(all_labels > 0, n_frames, n_frames + 1)
            else:
                all_pred = all_probs
                n_frames = all_pred.shape[1] if len(all_pred.shape) > 1 else 100
                time_of_accidents = np.where(all_labels > 0, n_frames, n_frames + 1)
            
            # 确保数组长度一致
            min_len = min(len(all_pred), len(all_labels), len(time_of_accidents))
            if min_len == 0:
                return 0.0  # 如果没有数据，返回0
            
            # 确保索引不会越界（evaluation函数内部可能使用len-1作为索引）
            if min_len < len(all_pred):
                all_pred = all_pred[:min_len]
            if min_len < len(all_labels):
                all_labels = all_labels[:min_len]
            if min_len < len(time_of_accidents):
                time_of_accidents = time_of_accidents[:min_len]
            
            # 确保time_of_accidents是1D数组
            if time_of_accidents.ndim > 1:
                time_of_accidents = time_of_accidents.flatten()
            
            try:
                AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, time_of_accidents, fps=fps)
                return AP
            except (IndexError, ValueError) as e:
                # 如果evaluation函数出错，返回0（表示性能不佳）
                return 0.0
        
        def compute_tta(all_labels, all_probs, fps=20.0):
            if len(all_probs.shape) == 1:
                n_videos = len(all_probs)
                n_frames = 100
                all_pred = np.tile(all_probs[:, None], (1, n_frames))
                time_of_accidents = np.where(all_labels > 0, n_frames, n_frames + 1)
            else:
                all_pred = all_probs
                time_of_accidents = np.where(all_labels > 0, all_pred.shape[1], all_pred.shape[1] + 1)
            AP, mTTA, TTA_R80, P_R80 = evaluation(all_pred, all_labels, time_of_accidents, fps=fps)
            return mTTA
    except ImportError as e:
        print(f"无法导入eval_tools: {e}")
        eval_tools_imported = False
        compute_ap = None
        compute_tta = None

# 导入DataLoader（从CRASH目录，创建VideoDataset适配器）
try:
    from src.DataLoader import DADDataset, CrashDataset, A3DDataset
    data_loader_imported = True
    
    # 创建VideoDataset适配器类
    class VideoDataset:
        def __init__(self, root_dir, dataset_name, split="train", feature="vgg16"):
            self.dataset_name = dataset_name.lower()
            self.split = split
            
            # 数据集路径映射
            dataset_map = {
                "dad": "dad",
                "crash": "crash", 
                "a3d": "a3d"
            }
            
            if self.dataset_name not in dataset_map:
                raise ValueError(f"未知数据集: {dataset_name}")
            
            data_path = os.path.join(root_dir, dataset_map[self.dataset_name])
            
            # 根据数据集类型选择对应的Dataset类和feature名称
            # DADDataset期望feature格式为"res101"或"vgg16"，对应目录为"res101_features"或"vgg16_features"
            # 但实际目录可能是"features"（对应res101）或"vgg16_features"
            if self.dataset_name == "dad":
                phase = "training" if split == "train" else "testing"  # DAD数据集使用"testing"而不是"test"
                # 检查实际存在的feature目录，并调整feature名称
                features_dir = os.path.join(data_path, "features")
                res101_features_dir = os.path.join(data_path, "res101_features")
                vgg16_features_dir = os.path.join(data_path, "vgg16_features")
                
                if os.path.exists(features_dir):
                    # 如果存在"features"目录，需要创建符号链接或使用特殊处理
                    # 但DADDataset期望{feature}_features格式，所以我们需要创建一个适配
                    # 最简单的方法是：如果feature是"res101"且只有"features"目录，使用"features"作为feature名
                    # 但DataLoader期望feature_name + "_features"，所以我们需要特殊处理
                    # 实际上，我们可以创建一个临时的符号链接，或者修改传入的feature名称
                    # 检查：如果features目录存在且res101_features不存在，可能需要特殊处理
                    if feature == "res101" and not os.path.exists(res101_features_dir):
                        # 创建临时符号链接（如果不存在）
                        if not os.path.exists(res101_features_dir):
                            try:
                                os.symlink(features_dir, res101_features_dir)
                            except (OSError, FileExistsError):
                                pass  # 符号链接已存在或创建失败
                    # 如果vgg16_features存在，也可以使用
                    if feature == "vgg16" and not os.path.exists(vgg16_features_dir) and os.path.exists(features_dir):
                        # 对于vgg16，如果不存在vgg16_features，可能需要使用其他feature
                        feature = "res101"  # 回退到res101
                
                self.dataset = DADDataset(data_path, feature, phase=phase, toTensor=False)
            elif self.dataset_name == "crash":
                phase = "train" if split == "train" else "test"
                self.dataset = CrashDataset(data_path, feature, phase=phase, toTensor=False)
            elif self.dataset_name == "a3d":
                phase = "train" if split == "train" else "test"
                self.dataset = A3DDataset(data_path, feature, phase=phase, toTensor=False)
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            # DADDataset返回(features, labels, toa)
            # 但labels可能是frame-level的(n_frames, 2)，需要转换为video-level的(2,)
            try:
                data = self.dataset[idx]
                if len(data) == 3:
                    features, labels, toa = data
                    # 如果labels是frame-level的(n_frames, 2)，转换为video-level
                    if isinstance(labels, np.ndarray) and len(labels.shape) == 2:
                        # labels是frame-level，检查是否有positive帧
                        has_positive = np.any(labels[:, 1] > 0) if labels.shape[1] > 1 else False
                        # 转换为video-level标签 [negative, positive]
                        labels = np.array([1 - int(has_positive), int(has_positive)], dtype=np.float32)
                    # 确保labels是numpy数组
                    if not isinstance(labels, np.ndarray):
                        labels = np.array(labels)
                    return features, labels
                else:
                    return data
            except ValueError as e:
                # 如果DataLoader内部出错，尝试修复
                if "ambiguous" in str(e):
                    # 这是DataLoader.py第58行的问题，需要修复数据
                    # 重新加载数据并修复labels格式
                    import sys
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CRASH"))
                    from src.DataLoader import DADDataset
                    # 直接访问底层数据
                    data_file = self.dataset.files_list[idx]
                    full_path = os.path.join(self.dataset.data_path, self.dataset.phase, data_file)
                    data = np.load(full_path)
                    features = data['data']
                    labels_frame = data['labels']
                    # 转换为video-level
                    has_positive = np.any(labels_frame[:, 1] > 0) if len(labels_frame.shape) > 1 and labels_frame.shape[1] > 1 else False
                    labels = np.array([1 - int(has_positive), int(has_positive)], dtype=np.float32)
                    return features, labels
                else:
                    raise
            
except ImportError as e:
    print(f"无法导入DataLoader: {e}")
    data_loader_imported = False
    VideoDataset = None

# 导入models_gru（尝试多个位置）
models_gru_imported = False
try:
    # 首先尝试从当前目录的src导入
    from src.models_gru import LFCRASH_CBM_GRU
    models_gru_imported = True
except ImportError:
    try:
        # 尝试直接导入
        import models_gru
        LFCRASH_CBM_GRU = models_gru.LFCRASH_CBM_GRU
        models_gru_imported = True
    except ImportError:
        # 尝试从CRASH目录导入（如果存在）
        try:
            sys.path.insert(0, crash_path)
            from src.models_gru import LFCRASH_CBM_GRU
            models_gru_imported = True
        except ImportError:
            # 最后尝试：检查是否有其他位置或pyc文件
            model_paths = [
                os.path.join(current_dir, "src", "models_gru.py"),
                os.path.join(current_dir, "models_gru.py"),
                os.path.join(crash_path, "src", "models_gru.py"),
            ]
            for path in model_paths:
                if os.path.exists(path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("models_gru", path)
                    models_gru_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(models_gru_module)
                    LFCRASH_CBM_GRU = models_gru_module.LFCRASH_CBM_GRU
                    models_gru_imported = True
                    break
            
            # 如果还没找到，尝试从pyc文件导入
            if not models_gru_imported:
                pyc_path = os.path.join(current_dir, "src", "__pycache__", "models_gru.cpython-312.pyc")
                if os.path.exists(pyc_path):
                    try:
                        import importlib.util
                        import marshal
                        with open(pyc_path, 'rb') as f:
                            f.read(16)  # 跳过pyc文件头
                            code = marshal.load(f)
                        models_gru_module = importlib.util.module_from_spec(importlib.util.spec_from_loader("models_gru", None))
                        exec(code, models_gru_module.__dict__)
                        LFCRASH_CBM_GRU = models_gru_module.LFCRASH_CBM_GRU
                        models_gru_imported = True
                    except Exception as e:
                        pass

if not all([models_gru_imported, data_loader_imported, eval_tools_imported]):
    print("=" * 80)
    print("导入错误！请检查以下模块：")
    print(f"  models_gru: {'✓' if models_gru_imported else '✗'}")
    print(f"  data_loader: {'✓' if data_loader_imported else '✗'}")
    print(f"  eval_tools: {'✓' if eval_tools_imported else '✗'}")
    print("=" * 80)
    
    # 提供更详细的错误信息
    if not models_gru_imported:
        print("\n找不到models_gru.py！请确认文件位置。")
        print("尝试查找的位置：")
        for path in [os.path.join(current_dir, "src", "models_gru.py"),
                     os.path.join(current_dir, "models_gru.py"),
                     os.path.join(crash_path, "src", "models_gru.py")]:
            print(f"  {path}: {'存在' if os.path.exists(path) else '不存在'}")
    
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_search_space(dataset_name):
    """
    为不同数据集返回不同的搜索空间
    基于之前的实验结果优化（根据DAD最佳结果和Crash当前最佳调整）
    """
    spaces = {
        "dad": {
            # DAD数据集：最佳AP=0.6511，最佳参数: λ_align=3.50e-05, λ_sparse=3.45e-04, batch_size=8, lr=4.44e-04, h_dim=512, z_dim=128
            # 围绕最佳参数扩展搜索空间
            "lambda_align": (1e-5, 1e-4),      # 最佳: 3.50e-05，缩小范围聚焦
            "lambda_sparse": (1e-4, 1e-3),     # 最佳: 3.45e-04，围绕最佳值
            "batch_size": [8, 16],            # 最佳: 8，重点关注小batch
            "learning_rate": (2e-4, 8e-4),    # 最佳: 4.44e-04，围绕最佳值
            "weight_decay": (1e-5, 1e-4),     # 最佳: 4.90e-05
            "h_dim": [256, 512, 768],         # 最佳: 512
            "z_dim": [128, 256],              # 最佳: 128，重点关注小维度
        },
        "crash": {
            # Crash数据集：当前最佳AP=0.6667，搜索空间围绕已有最佳结果
            "lambda_align": (1e-4, 2e-3),     # 围绕0.0008附近
            "lambda_sparse": (1e-3, 3e-2),     # 围绕0.019附近
            "batch_size": [8, 16, 32],        # 保持多样性
            "learning_rate": (5e-5, 5e-4),    # 围绕0.0002附近
            "weight_decay": (1e-5, 1e-4),
            "h_dim": [256, 512, 768],         # 最佳: 512
            "z_dim": [128, 256, 512],         # 最佳: 256
        },
        "a3d": {
            # A3D数据集：基于DAD和Crash的经验，使用中等范围
            "lambda_align": (1e-5, 1e-3),     # 扩大范围探索
            "lambda_sparse": (1e-4, 2e-2),    # 扩大范围探索
            "batch_size": [8, 16, 32],
            "learning_rate": (1e-5, 5e-4),    # 扩大范围探索
            "weight_decay": (1e-6, 1e-4),
            "h_dim": [256, 512, 768],
            "z_dim": [128, 256, 512],
        }
    }
    
    if dataset_name.lower() not in spaces:
        logger.warning(f"数据集 {dataset_name} 未定义搜索空间，使用默认DAD空间")
        return spaces["dad"]
    
    return spaces[dataset_name.lower()]


def train_one_trial(trial, dataset_name, n_epochs=15, device="cuda", num_workers=2):
    """
    训练一个trial
    """
    try:
        # 获取搜索空间
        space = get_search_space(dataset_name)
        
        # 采样超参数
        lambda_align = trial.suggest_float("lambda_align", space["lambda_align"][0], space["lambda_align"][1], log=True)
        lambda_sparse = trial.suggest_float("lambda_sparse", space["lambda_sparse"][0], space["lambda_sparse"][1], log=True)
        batch_size = trial.suggest_categorical("batch_size", space["batch_size"])
        learning_rate = trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True)
        weight_decay = trial.suggest_float("weight_decay", space["weight_decay"][0], space["weight_decay"][1], log=True)
        h_dim = trial.suggest_categorical("h_dim", space["h_dim"])
        z_dim = trial.suggest_categorical("z_dim", space["z_dim"])
        
        logger.info(f"Trial {trial.number}: λ_align={lambda_align:.6f}, λ_sparse={lambda_sparse:.6f}, "
                   f"batch_size={batch_size}, lr={learning_rate:.6f}, h_dim={h_dim}, z_dim={z_dim}")
        sys.stdout.flush()
        
        # 加载数据集
        logger.info(f"Trial {trial.number}: 开始加载数据集...")
        sys.stdout.flush()
        
        # 数据集路径（尝试多个可能的位置）
        possible_data_roots = [
            os.path.join(os.path.dirname(__file__), "data"),
            os.path.join(os.path.dirname(__file__), "..", "CRASH", "data"),
            os.path.join(os.path.dirname(__file__), "..", "data"),
            "/data/sony/LFCRASH/CRASH/data",  # 绝对路径
            "/data/sony/LFCRASH/data",  # 另一个可能的绝对路径
        ]
        
        data_root = None
        for root in possible_data_roots:
            test_path = os.path.join(root, dataset_name if dataset_name != "crash" else "crash")
            if os.path.exists(test_path):
                data_root = root
                break
        
        if data_root is None:
            raise ValueError(f"找不到数据集路径！尝试的位置: {possible_data_roots}")
        
        logger.info(f"Trial {trial.number}: 使用数据路径: {data_root}")
        sys.stdout.flush()
        
        # 根据数据集确定feature类型（所有数据集都使用vgg16）
        feature_type = "vgg16"
        
        train_dataset = VideoDataset(
            root_dir=data_root,
            dataset_name=dataset_name,
            split="train",
            feature=feature_type
        )
        test_dataset = VideoDataset(
            root_dir=data_root,
            dataset_name=dataset_name,
            split="test",
            feature=feature_type
        )
        
        logger.info(f"Trial {trial.number}: 数据集加载完成，训练集{len(train_dataset)}样本，测试集{len(test_dataset)}样本")
        sys.stdout.flush()
        
        # 创建DataLoader（优化内存使用）
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None,
                drop_last=True
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None
            )
            logger.info(f"Trial {trial.number}: 数据加载器创建成功 (num_workers={num_workers})")
        except Exception as e:
            logger.warning(f"Trial {trial.number}: 多进程数据加载失败，回退到单进程: {e}")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        sys.stdout.flush()
        
        # 初始化模型
        logger.info(f"Trial {trial.number}: 开始初始化模型...")
        sys.stdout.flush()
        
        concept_file = os.path.join(os.path.dirname(__file__), "..", "000_all_concept_set.txt")
        if not os.path.exists(concept_file):
            concept_file = None
        
        # 根据数据集设置模型参数（使用vgg16特征，维度为4096）
        dataset_params = {
            "dad": {"x_dim": 4096, "n_obj": 19, "n_frames": 100, "fps": 20.0},
            "crash": {"x_dim": 4096, "n_obj": 19, "n_frames": 50, "fps": 10.0},
            "a3d": {"x_dim": 4096, "n_obj": 19, "n_frames": 100, "fps": 20.0},
        }
        params = dataset_params.get(dataset_name.lower(), dataset_params["dad"])
        
        model = LFCRASH_CBM_GRU(
            x_dim=params["x_dim"],
            h_dim=h_dim,
            z_dim=z_dim,
            n_layers=2,  # 默认值
            n_obj=params["n_obj"],
            n_frames=params["n_frames"],
            fps=params["fps"],
            with_saa=True,  # 默认值
            num_concepts=837,
            concept_file=concept_file,
            lambda_align=lambda_align,
            lambda_sparse=lambda_sparse,
            device=device
        ).to(device)
        
        logger.info(f"Trial {trial.number}: 模型初始化完成，开始训练...")
        sys.stdout.flush()
        
        # 优化器和调度器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_ap = 0.0
        
        # 训练循环
        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            logger.info(f"Trial {trial.number}: Epoch {epoch}/{n_epochs}")
            sys.stdout.flush()
            
            for batch_idx, (x, y) in enumerate(train_loader):
                # 数据转移到GPU（非阻塞）
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True) if y is not None else None
                
                # 从y中提取toa（如果需要）
                # DADDataset返回(features, labels, toa)，但我们已经转换为(features, labels)
                # 需要从原始数据中获取toa，或者根据labels计算
                if y is not None:
                    # 如果y是video-level标签，需要构造toa
                    # 对于positive样本，toa通常是某个固定值或需要从数据中获取
                    # 这里简化处理：positive样本toa=90（DAD数据集），negative样本toa=n_frames+1
                    batch_size = x.size(0)
                    toa = []
                    for i in range(batch_size):
                        if len(y.shape) == 2:
                            label = y[i, 1] if y.shape[1] > 1 else y[i, 0]
                        else:
                            label = y[i] if len(y.shape) == 1 else y[i, 1]
                        if label > 0.5:  # positive
                            toa.append([90.0])  # DAD数据集默认toa
                        else:  # negative
                            toa.append([params["n_frames"] + 1])
                    toa = torch.tensor(toa, dtype=torch.float32, device=device)
                else:
                    toa = None
                
                optimizer.zero_grad()
                
                # 前向传播（需要toa参数）
                if toa is not None:
                    losses, frame_logits, concept_activations, frame_probs = model(x, y, toa)
                else:
                    losses, frame_logits, concept_activations, frame_probs = model(x, None, None)
                total_loss = losses["total"]
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                n_batches += 1
            
            scheduler.step()
            
            # 每5个epoch评估一次（节省时间）
            if epoch % 5 == 0 or epoch == n_epochs:
                model.eval()
                all_probs = []
                all_labels = []
                
                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True) if y is not None else None
                        
                        # 构造toa（测试时）
                        if y is not None:
                            batch_size = x.size(0)
                            toa = []
                            for i in range(batch_size):
                                if len(y.shape) == 2:
                                    label = y[i, 1] if y.shape[1] > 1 else y[i, 0]
                                else:
                                    label = y[i] if len(y.shape) == 1 else y[i, 1]
                                if label > 0.5:
                                    toa.append([90.0])
                                else:
                                    toa.append([params["n_frames"] + 1])
                            toa = torch.tensor(toa, dtype=torch.float32, device=device)
                        else:
                            toa = None
                        
                        _, _, _, frame_probs = model(x, None, toa)
                        video_probs = frame_probs.mean(dim=1).cpu().numpy()
                        
                        if y is not None:
                            video_labels = y[:, 0].cpu().numpy()
                            all_probs.extend(video_probs)
                            all_labels.extend(video_labels)
                
                if len(all_probs) > 0:
                    # 确保数组长度一致
                    all_labels_array = np.array(all_labels)
                    all_probs_array = np.array(all_probs)
                    min_len = min(len(all_labels_array), len(all_probs_array))
                    if min_len > 0:
                        all_labels_array = all_labels_array[:min_len]
                        all_probs_array = all_probs_array[:min_len]
                        ap_video = compute_ap(all_labels_array, all_probs_array, fps=params["fps"], n_frames=params["n_frames"])
                        best_ap = max(best_ap, ap_video)
                    
                    logger.info(f"Trial {trial.number}: Epoch {epoch} AP_video={ap_video:.4f}, best_AP={best_ap:.4f}")
                    sys.stdout.flush()
                    
                    # 报告中间结果给Optuna
                    trial.report(ap_video, epoch)
                    
                    # 早停检查
                    if trial.should_prune():
                        logger.info(f"Trial {trial.number}: 被剪枝")
                        sys.stdout.flush()
                        raise optuna.TrialPruned()
        
        logger.info(f"Trial {trial.number} completed: best_AP={best_ap:.4f}")
        sys.stdout.flush()
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        return best_ap
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
        sys.stdout.flush()
        torch.cuda.empty_cache()
        raise


def objective(trial, dataset_name, n_epochs, device, num_workers):
    """Optuna目标函数"""
    return train_one_trial(trial, dataset_name, n_epochs, device, num_workers)


def main():
    parser = argparse.ArgumentParser(description="多数据集Optuna超参数搜索")
    parser.add_argument("--dataset", type=str, required=True, choices=["dad", "crash", "a3d"],
                       help="数据集名称")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--n_trials", type=int, default=20, help="试验次数")
    parser.add_argument("--n_epochs", type=int, default=15, help="每个trial的训练轮数")
    parser.add_argument("--study_name", type=str, default=None, help="Study名称（可选）")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载器工作进程数")
    
    args = parser.parse_args()
    
    # 设置设备
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(f"optuna_studies/gpu{args.gpu_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Study名称
    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"optuna_gpu{args.gpu_id}_{args.dataset}_{timestamp}"
    else:
        study_name = args.study_name
    
    # 创建Study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    # 日志文件
    log_file = logs_dir / f"{study_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"开始Optuna搜索: 数据集={args.dataset}, GPU={args.gpu_id}, Trials={args.n_trials}, Epochs={args.n_epochs}")
    logger.info(f"搜索空间: {get_search_space(args.dataset)}")
    sys.stdout.flush()
    
    # 运行优化
    try:
        study.optimize(
            lambda trial: objective(trial, args.dataset, args.n_epochs, device, args.num_workers),
            n_trials=args.n_trials,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("搜索被用户中断")
    
    # 保存结果
    results_file = output_dir / f"{study_name}_results.json"
    results = {
        "study_name": study_name,
        "dataset": args.dataset,
        "n_trials": args.n_trials,
        "n_epochs": args.n_epochs,
        "best_trial": study.best_trial.number if study.best_trials else None,
        "best_value": study.best_value if study.best_trials else None,
        "best_params": study.best_params if study.best_trials else None,
        "n_complete": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == TrialState.PRUNED]),
        "n_fail": len([t for t in study.trials if t.state == TrialState.FAIL]),
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("搜索完成!")
    logger.info(f"最佳Trial: {results['best_trial']}")
    logger.info(f"最佳AP: {results['best_value']:.4f}")
    logger.info(f"最佳参数: {results['best_params']}")
    logger.info(f"完成: {results['n_complete']}, 剪枝: {results['n_pruned']}, 失败: {results['n_fail']}")
    logger.info("=" * 80)
    sys.stdout.flush()
    
    print(f"\n结果已保存到: {results_file}")
    print(f"日志已保存到: {log_file}")


if __name__ == "__main__":
    main()

