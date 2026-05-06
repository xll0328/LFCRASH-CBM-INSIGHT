#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试Optuna脚本的导入是否正常"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("测试导入...")

try:
    from src.models_gru import LFCRASH_CBM_GRU
    print("✓ models_gru 导入成功")
except ImportError as e:
    print(f"✗ models_gru 导入失败: {e}")

try:
    from src.data_loader import VideoDataset
    print("✓ data_loader 导入成功")
except ImportError as e:
    print(f"✗ data_loader 导入失败: {e}")

try:
    from src.eval_tools import compute_ap, compute_tta
    print("✓ eval_tools 导入成功")
except ImportError as e:
    print(f"✗ eval_tools 导入失败: {e}")

print("\n如果所有导入都成功，可以运行Optuna搜索了！")




