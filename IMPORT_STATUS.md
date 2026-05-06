# 导入问题说明

当前Optuna搜索启动失败，原因是缺少以下模块：
- `models_gru.py` (包含 LFCRASH_CBM_GRU 类)
- `data_loader.py` 或 `VideoDataset` 类
- `eval_tools.py` (包含 compute_ap, compute_tta 函数)

## 已找到的文件
- `CRASH/src/eval_tools.py` ✓
- `CRASH/src/DataLoader.py` ✓ (但类名是 DADDataset，不是 VideoDataset)

## 需要确认
1. `models_gru.py` 文件的实际位置
2. `VideoDataset` 类的实际位置和名称
3. 之前成功的实验使用的实际导入路径

## 建议
请检查之前的实验日志，确认实际使用的训练脚本和导入路径。
或者，如果这些文件在其他位置，请告知具体路径。
