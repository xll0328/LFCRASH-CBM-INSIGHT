# LFCRASH-CBM 框架文档

## 一、项目概述

**LFCRASH-CBM** 是一个结合了 **CRASH** 视频异常检测架构和 **Label-Free CBM** 概念可解释性的混合模型。该框架在保持CRASH完整架构的基础上，引入了概念瓶颈模型（Concept Bottleneck Model）的思想，使模型不仅能够检测异常，还能提供可解释的概念激活。

### 核心思想

1. **保持CRASH架构完整性**：完全保留CRASH的所有核心模块（SAA、RSD、FFT、GRU等）
2. **引入概念投影层**：在CRASH特征提取后，添加概念投影层，将特征映射到概念空间
3. **Label-Free训练**：使用CLIP编码的概念文本嵌入，无需人工标注概念标签

---

## 二、整体架构

### 2.1 数据流

```
输入特征 (B, T, N, D) 或 (B, T, D)
    ↓
维度处理 (聚合对象/高度维度)
    ↓
phi_x: 线性投影 (D → 512)
    ↓
SpatialAttention (可选): 空间注意力
    ↓
RSD层: 时序关系建模
    ↓
FFT Block: 频域特征增强
    ↓
【概念投影层】← 新增
    ↓
概念激活 (B, T, 837)
    ↓
GRU网络: 时序建模
    ↓
SelfAttAggregate: 自注意力聚合
    ↓
分类器: 视频级和帧级预测
```

### 2.2 核心组件

#### 1. **phi_x** - 特征提取层
- **作用**: 将输入特征维度统一到512维
- **输入**: (B, T, D) - D可以是2048（res101）或4096（vgg16）
- **输出**: (B, T, 512)

#### 2. **SpatialAttention (SAA)** - 空间注意力
- **作用**: 增强空间特征表示
- **实现**: 深度可分离卷积 + Sigmoid激活
- **输入**: (B*T, 512, 1, 1)
- **输出**: (B, T, 512)

#### 3. **RSD层** - 关系序列解码器
- **来源**: CRASH原始实现 (`CRASH/src/RSDlayerAttention.py`)
- **作用**: 建模时序关系，使用多头注意力机制
- **输入**: (B, 2, T, 512)
- **输出**: (2, T, 512) → 平均后 (B, T, 512)

#### 4. **FFT Block** - 频域特征增强
- **来源**: CRASH原始实现 (`CRASH/src/fft.py`)
- **作用**: 在频域进行特征增强
- **输入**: (B*T, 512, 1)
- **输出**: (B, T, 512)

#### 5. **概念投影层** - Concept Projection Layer ⭐
- **作用**: 将CRASH特征投影到概念空间
- **输入**: (B, T, 512)
- **输出**: (B, T, 837) - 837个概念激活
- **实现**: `nn.Linear(512, num_concepts)`

#### 6. **GRUNet** - 时序建模网络
- **作用**: 对概念激活序列进行时序建模
- **输入**: (B, T, 1861) = 概念激活(837) + 零填充(1024)
- **内部**: GRU会额外添加512维，实际输入为2373维
- **输出**: (B, T, z_dim) - 默认256维

#### 7. **SelfAttAggregate** - 自注意力聚合
- **作用**: 将帧级特征聚合为视频级特征
- **实现**: MultiheadAttention + LayerNorm
- **输入**: (B, T, z_dim)
- **输出**: (B, z_dim)

#### 8. **分类器** - Classifier
- **作用**: 最终分类预测
- **输入**: 视频级特征 (B, z_dim) 或 帧级特征 (B, T, z_dim)
- **输出**: (B, 2) 或 (B, T, 2)

---

## 三、损失函数设计

### 3.1 主损失 (Main Loss)
- **类型**: 视频级交叉熵损失
- **公式**: `CrossEntropy(video_logit, y)`
- **作用**: 优化视频级分类性能

### 3.2 辅助损失 (Auxiliary Loss)
- **类型**: 帧级时间加权交叉熵损失
- **公式**: `WeightedCrossEntropy(frame_logits, frame_labels, weights)`
- **权重**: 对于positive样本，越接近事故时间(toa)的帧权重越大
  - `weight[t] = 1.0 + (toa - t) / toa`
- **作用**: 增强对事故时刻附近帧的预测能力

### 3.3 概念对齐损失 (Concept Alignment Loss) ⭐
- **类型**: MSE损失
- **公式**: `MSE(similarity, concept_act_norm)`
  - `similarity = normalize(crash_feat) @ normalize(concept_embeddings).T`
  - `concept_act_norm = normalize(concept_activations.mean(dim=1))`
- **作用**: 对齐CRASH特征与CLIP概念嵌入的相似度，确保概念激活与语义概念一致
- **权重**: `lambda_align` (默认0.001)

### 3.4 稀疏性损失 (Sparsity Loss) ⭐
- **类型**: L1正则化
- **公式**: `mean(abs(concept_activations))`
- **作用**: 鼓励概念激活稀疏，提高可解释性
- **权重**: `lambda_sparse` (默认0.01)

### 3.5 总损失
```
total_loss = main_loss + aux_loss + lambda_align * align_loss + lambda_sparse * sparse_loss
```

---

## 四、概念系统

### 4.1 概念来源
- **概念文件**: `/data/sony/LFCRASH/000_all_concept_set.txt`
- **概念数量**: 837个
- **概念类型**: 与交通场景相关的语义概念（如"car", "road", "accident"等）

### 4.2 CLIP编码
- **模型**: CLIP ViT-B/16
- **编码方式**: 使用CLIP的文本编码器将概念文本编码为512维向量
- **归一化**: L2归一化，确保嵌入在单位球面上
- **数据类型**: float32（避免与模型其他部分的float16冲突）

### 4.3 概念对齐机制
1. **CRASH特征归一化**: 将每帧的CRASH特征(512维)归一化
2. **计算相似度**: `similarity = crash_feat_norm @ concept_embeddings.T` → (B, 837)
3. **概念激活归一化**: 将概念激活归一化
4. **对齐损失**: 使用MSE损失对齐相似度和概念激活

---

## 五、数据集支持

### 5.1 DAD数据集
- **特征维度**: 4096 (vgg16)
- **输入格式**: (B, T, N, H, D) = (B, 10, 100, 20, 4096)
- **处理**: 聚合对象和高度维度 → (B, 10, 4096)
- **帧数**: 100
- **FPS**: 20.0
- **特殊处理**: 
  - labels是frame-level (10, 2)，需要转换为video-level
  - 使用DADDatasetWrapper包装器

### 5.2 Crash数据集
- **特征维度**: 4096 (vgg16)
- **输入格式**: (B, T, N, D) = (B, 50, 19, 4096)
- **处理**: 聚合对象维度 → (B, 50, 4096)
- **帧数**: 50
- **FPS**: 10.0

### 5.3 A3D数据集
- **特征维度**: 4096 (vgg16)
- **输入格式**: (B, T, N, D) = (B, 100, 19, 4096)
- **处理**: 聚合对象维度 → (B, 100, 4096)
- **帧数**: 100
- **FPS**: 20.0

---

## 六、训练流程

### 6.1 训练脚本
- **主脚本**: `train_best_params.py`
- **启动脚本**: `run_best_params_gpu6_parallel.sh`
- **模式**: 三个数据集并行训练

### 6.2 训练参数（最佳配置）
所有数据集使用相同的超参数：
- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Epochs**: 25
- **Lambda Align**: 0.001
- **Lambda Sparse**: 0.01
- **Weight Decay**: 1e-5
- **h_dim**: 512
- **z_dim**: 256

### 6.3 训练步骤
1. **数据加载**: 使用CRASH的DataLoader，支持DAD、Crash、A3D
2. **模型初始化**: 加载CLIP模型，编码837个概念
3. **训练循环**: 
   - 前向传播 → 计算损失 → 反向传播 → 更新参数
   - 每个epoch后评估
4. **模型保存**: 保存最佳AP对应的模型

---

## 七、当前训练状态

### 7.1 部署信息
- **GPU**: 6 (NVIDIA GeForce RTX 4090, 24GB)
- **模式**: 并行训练（三个数据集同时运行）
- **进程**:
  - DAD: PID 1643228
  - Crash: PID 1643229
  - A3D: PID 1643231

### 7.2 训练进度
- **DAD**: 已完成第1个epoch，正在第2个epoch
- **Crash**: 正在第1个epoch
- **A3D**: 已完成第2个epoch，正在第3个epoch

### 7.3 预期性能（基于最佳参数）
- **DAD**: AP (video) = 0.5229, AP (frame) = 0.5068, mTTA = 3.5568
- **Crash**: AP (video) = 0.9977, AP (frame) = 0.9636, mTTA = 4.7286
- **A3D**: AP (video) = 0.9403, AP (frame) = 0.9408, mTTA = 3.6605

---

## 八、关键技术细节

### 8.1 维度处理
- **输入维度适配**: 支持3D、4D、5D输入，自动聚合多余维度
- **特征维度统一**: 通过phi_x将不同输入维度统一到512维
- **GRU输入处理**: 概念激活(837) + 零填充(1024) = 1861维，GRU内部再添加512维

### 8.2 数据类型处理
- **统一为float32**: 所有tensor统一为float32，避免Half和Float类型不匹配
- **CLIP embeddings**: 加载时立即转换为float32

### 8.3 toa处理
- **统一为1D tensor**: (B,) 格式
- **处理多种输入**: list、numpy array、tensor等

### 8.4 DAD数据集特殊处理
- **labels转换**: frame-level (10, 2) → video-level (2,)
- **直接读取数据文件**: 避免CRASH DataLoader的labels[1] > 0错误

---

## 九、与CRASH和Label-free-CBM的关系

### 9.1 与CRASH的关系
- **继承**: 完全保留CRASH的所有核心模块
  - phi_x特征提取
  - SpatialAttention (SAA)
  - RSD层（Encoder）
  - FFT Block
  - GRU网络
  - SelfAttAggregate
- **扩展**: 在CRASH特征提取后添加概念投影层

### 9.2 与Label-free-CBM的关系
- **概念投影**: 借鉴CBM的proj_layer思想
- **CLIP对齐**: 使用CLIP文本嵌入进行概念对齐
- **稀疏性**: 使用L1正则化鼓励概念激活稀疏
- **Label-Free**: 无需人工标注概念标签，使用CLIP自动编码

### 9.3 创新点
1. **在视频异常检测中引入概念可解释性**
2. **保持CRASH完整架构的同时添加概念层**
3. **使用CLIP进行Label-Free的概念对齐**

---

## 十、文件结构

```
LFCRASH-CBM/
├── src/
│   ├── models_gru.py          # 主模型定义
│   ├── data_loader.py          # 数据加载（导入CRASH的DataLoader）
│   └── eval_tools.py           # 评估工具（导入CRASH的evaluation）
├── train_best_params.py       # 训练脚本
├── train_optuna_multi_dataset.py  # Optuna超参数搜索
├── run_best_params_gpu6_parallel.sh  # GPU 6并行训练启动脚本
├── best_params.json           # 最佳参数配置
├── output_gru/                # 模型输出目录
│   └── best_params_gpu6/      # GPU 6训练结果
└── logs/                      # 训练日志
    └── best_params_gpu6/      # GPU 6训练日志
```

---

## 十一、监控和调试

### 11.1 日志查看
```bash
# 查看所有日志
tail -f logs/best_params_gpu6/train_*.log

# 查看特定数据集
tail -f logs/best_params_gpu6/train_dad_*.log
tail -f logs/best_params_gpu6/train_crash_*.log
tail -f logs/best_params_gpu6/train_a3d_*.log
```

### 11.2 GPU监控
```bash
nvidia-smi -i 6 -l 1
```

### 11.3 进程监控
```bash
ps aux | grep train_best_params | grep -v grep
```

---

## 十二、未来改进方向

1. **概念选择优化**: 根据数据集特点选择更相关的概念
2. **概念对齐策略**: 探索更有效的对齐方法
3. **多尺度概念**: 支持不同粒度的概念（细粒度、粗粒度）
4. **概念可视化**: 开发工具可视化概念激活
5. **性能优化**: 进一步优化GPU利用率

---

## 附录：关键代码位置

- **模型定义**: `src/models_gru.py` - `LFCRASH_CBM_GRU`类
- **前向传播**: `src/models_gru.py` - `forward`方法（第196-315行）
- **损失计算**: `src/models_gru.py` - `_compute_losses`方法（第317-426行）
- **训练循环**: `train_best_params.py` - `train_dataset`函数
- **数据加载**: `src/data_loader.py` - 导入CRASH的DataLoader
- **评估工具**: `src/eval_tools.py` - 导入CRASH的evaluation函数




