# CRASH对齐训练部署文档

**部署时间**: 2026-01-10 17:08  
**GPU**: 7  
**架构**: CRASH对齐后的LFCRASH-CBM-GRU

## ✅ 已完成的CRASH组件对齐

1. **SpatialAttention (OFA)** - Object Focus Attention
   - 使用双层GRU隐藏状态作为query
   - 对象嵌入作为key/value

2. **PositionalEncoding** - 位置编码模块
   - 用于TFA组件，支持动态维度

3. **SelfAttAggregate (TFA)** - Temporal Focus Attention
   - 带位置编码的自注意力聚合
   - maxpool + avgpool拼接

4. **AccidentPredictor** - 辅助预测器
   - 用于SAA辅助损失

5. **GRUNet** - 匹配CRASH实现
   - 输入维度: h_dim+h_dim+512 = 1536
   - 输出维度: 2 (二分类logits)

6. **_exp_loss** - 指数损失函数
   - 时间加权的指数惩罚

7. **维度适配修复**
   - FFT维度适配（h_dim -> 512投影）
   - RSD维度适配（h_dim -> 512投影）
   - PositionalEncoding维度适配（使用agg_dim）
   - Concept投影维度适配（使用h_dim）

## 📊 训练配置

### DAD数据集
- Epochs: 50
- Batch Size: 8
- Learning Rate: 3.99e-04
- h_dim: 256
- z_dim: 128

### Crash数据集
- Epochs: 50
- Batch Size: 16
- Learning Rate: 2.40e-04
- h_dim: 512
- z_dim: 256

### A3D数据集（重点提升）
- Epochs: 50
- Batch Size: 32
- Learning Rate: 1.04e-05
- h_dim: 768
- z_dim: 256

## 🎯 目标

- **A3D AP > 0.3** (当前0.0879，需要提升3倍以上)
- 完整评估指标（AP, mTTA, TTA@R80, P@R80）
- CRASH组件对齐验证

## 📝 文件位置

- 日志: `logs/crash_aligned_training_20260110_170833/`
- 输出: `output/crash_aligned_training_20260110_170833/`
- 部署脚本: `deploy_crash_aligned_training.sh`

## 🔍 监控命令

```bash
# 查看进程状态
ps aux | grep 'train_best_params.py' | grep -v grep

# 查看GPU状态
nvidia-smi -i 7 -l 1

# 查看日志（实时）
tail -f logs/crash_aligned_training_20260110_170833/train_*.log

# 停止所有实验
pkill -f 'train_best_params.py'
```

## ⏱️ 预计完成时间

约24小时后（2026-01-11下午）

