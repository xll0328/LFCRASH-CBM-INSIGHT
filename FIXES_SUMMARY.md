# 修复总结

## 已修复的问题

### 1. DAD数据集labels处理 ✅
- **问题**: DAD的labels是(10, 2)的frame-level数组，CRASH原始代码`labels[1] > 0`会报错
- **修复**: 创建DADDatasetWrapper，直接读取数据文件，将frame-level labels转换为video-level
- **位置**: `train_best_params.py` 第146-186行

### 2. 输入维度处理 ✅
- **问题**: DAD数据集输入是5维 (B, T, N, H, D)
- **修复**: 在forward方法中处理5维输入，聚合对象和高度维度
- **位置**: `src/models_gru.py` 第224-235行

### 3. CLIP embeddings类型 ✅
- **问题**: CLIP返回float16，计算时类型不匹配
- **修复**: 在加载时立即转换为float32
- **位置**: `src/models_gru.py` 第183行

### 4. 概念对齐损失 ✅
- **优化**: 参考Label-free-CBM的实现，对齐crash特征与CLIP概念嵌入的相似度
- **位置**: `src/models_gru.py` 第370-400行

### 5. toa维度处理 ✅
- **修复**: 统一处理为1D tensor (B,)
- **位置**: `train_best_params.py` 多处，`src/models_gru.py` 第343-356行

## 部署信息

- **GPU**: 2 (NVIDIA GeForce RTX 4090)
- **启动脚本**: `run_best_params_gpu2.sh`
- **输出目录**: `output_gru/best_params_gpu2/`
- **日志目录**: `logs/best_params_gpu2/`

## 监控命令

```bash
tail -f logs/best_params_gpu2/train_*.log
nvidia-smi -i 2 -l 1
ps aux | grep train_best_params
```

