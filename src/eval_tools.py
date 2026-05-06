# -*- coding: utf-8 -*-
"""
评估工具 - 从CRASH目录导入
"""

import sys
import os

# 添加CRASH目录到路径
crash_path = os.path.join(os.path.dirname(__file__), '..', '..', 'CRASH')
if os.path.exists(crash_path):
    sys.path.insert(0, crash_path)
    from src.eval_tools import evaluation, evaluation_train, evaluation_P_R80, print_results
    __all__ = ['evaluation', 'evaluation_train', 'evaluation_P_R80', 'print_results', 'compute_ap', 'compute_tta']
    
    # 包装函数以匹配train_optuna_multi_dataset.py的期望
    def compute_ap(all_pred, all_labels, time_of_accidents, fps=20.0):
        """计算AP（Average Precision）"""
        AP, _, _, _ = evaluation(all_pred, all_labels, time_of_accidents, fps)
        return AP
    
    def compute_tta(all_pred, all_labels, time_of_accidents, fps=20.0):
        """计算mTTA（mean Time to Accident）"""
        _, mTTA, _, _ = evaluation(all_pred, all_labels, time_of_accidents, fps)
        return mTTA
else:
    raise ImportError(f"无法找到CRASH目录: {crash_path}")
