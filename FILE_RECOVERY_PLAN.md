# 文件恢复计划

## 缺失文件分析

### 确认缺失的关键文件：
1. **src/models_gru.py** - 核心模型文件（31KB pyc存在）
2. **train_gru.py** - 训练脚本（日志显示之前使用过）
3. **src/data_loader.py** - 数据加载器（16KB pyc存在）
4. **src/eval_tools.py** - 评估工具（5.3KB pyc存在）

### 存在的pyc文件（可反编译）：
- models_gru.cpython-312.pyc (31KB)
- data_loader.cpython-312.pyc (16KB)  
- eval_tools.cpython-312.pyc (5.3KB)
- models.cpython-312.pyc (23KB)
- visualization.cpython-312.pyc (20KB)

## 恢复方案

### 方案1：从pyc文件反编译（推荐）
使用uncompyle6或decompyle3工具反编译pyc文件恢复源代码

### 方案2：从git历史恢复
检查git历史，看是否有这些文件的提交记录

### 方案3：基于现有代码重建
基于CRASH目录的代码和实验日志重建缺失文件

## 项目背景
- LFCRASH-CBM：结合CRASH架构和Label-Free CBM的视频异常检测
- 数据集：DAD、CCD(crash)、A3D
- 之前实验成功运行，结果接近CRASH原文
