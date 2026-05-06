# -*- coding: utf-8 -*-
"""
数据加载器 - 从CRASH目录导入
"""

import sys
import os

# 添加CRASH目录到路径
crash_path = os.path.join(os.path.dirname(__file__), '..', '..', 'CRASH')
if os.path.exists(crash_path):
    sys.path.insert(0, crash_path)
    from src.DataLoader import DADDataset, CrashDataset, A3DDataset
    __all__ = ['DADDataset', 'CrashDataset', 'A3DDataset']
else:
    raise ImportError(f"无法找到CRASH目录: {crash_path}")
