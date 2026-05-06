# CRASH代码对齐检查清单

## 📋 目的
在debug时优先参考CRASH-main中的原始实现，确保LFCRASH-CBM与CRASH框架对齐。

## 🔍 关键对齐点

### 1. 模型forward方法返回值
**CRASH原始实现** (`CRASH-main/src/Models.py:282`):
```python
return losses, all_outputs, all_hidden
```

**对齐检查**:
- ✅ `losses`: 字典，包含 `'cross_entropy'`, `'total_loss'`, `'log'`, 以及可选的 `'auxloss'`
- ✅ `all_outputs`: 列表，每个元素是 `(B, 2)` 的logits（未经过softmax）
- ✅ `all_hidden`: 列表，每个元素是最后一层的隐藏状态 `h[-1]`

### 2. Loss计算方式
**CRASH原始实现** (`CRASH-main/src/Models.py:268-292`):
```python
L3 = self._exp_loss(output, y, t, toa=toa, fps=self.fps)
losses['cross_entropy'] += L3  # 累加每个时间步的损失

# SAA辅助损失
if self.with_saa:
    embed_video = self.self_aggregation(torch.stack(all_hidden, dim=-1))
    dec = self.predictor_aux(embed_video)
    L4 = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))
    L4 = L4 / (self.rho_2 * self.rho_2)
    losses['auxloss'] = L4

losses['log'] = torch.log(self.rho_1 * self.rho_2)
```

**对齐检查**:
- ✅ `cross_entropy` 是累加的（每个时间步都加）
- ✅ `auxloss` 只在最后计算一次
- ✅ `log` 项是 `torch.log(self.rho_1 * self.rho_2)`
- ✅ `total_loss` 在forward中初始化为0，通常在主训练循环中计算

### 3. RSD (Recurrent Self-Attention) 使用方式
**CRASH原始实现** (`CRASH-main/src/Models.py:242-265`):
```python
h_list.append(h)

if t==2:
    h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2]),dim=0)
    h = self.RSD(h_staked)
elif t==3:
    h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3]),dim=0)
    h = self.RSD(h_staked)
# ... 类似地处理 t==4,5,6,7,8
elif t>8:
    h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3], 
                           h_list[t-4], h_list[t-5], h_list[t-6], h_list[t-7], 
                           h_list[t-8], h_list[t-9]),dim=0)
    h = self.RSD(h_staked)
```

**对齐检查**:
- ✅ RSD从 `t>=2` 开始使用
- ✅ 使用最近的历史隐藏状态（最多10个）
- ✅ RSD输入形状：`(n_layers, B, h_dim)`，输出形状：`(n_layers, B, h_dim)`

### 4. 特征处理流程
**CRASH原始实现** (`CRASH-main/src/Models.py:220-239`):
```python
x_t = self.phi_x(x[:, t])  # (B, N+1, h_dim)

img_embed = x_t[:, 0, :].unsqueeze(1)  # (B, 1, h_dim)
img_tmp = img_embed.view(x.size(0), 512, 1)  # 假设h_dim=512
img_tmp = self.fftblock(img_tmp)  # FFT处理
img_tmp = img_tmp.view(x.size(0), 1, 512)
img_fft = self.phi_x3(img_tmp)  # (B, 1, 512)

obj_embed = x_t[:, 1:, :]  # (B, n_obj, h_dim)
obj_embed = self.sp_attention(obj_embed, h)  # (B, 1, h_dim)

x_t = torch.cat([obj_embed, img_embed, img_fft], dim=-1)  # (B, 1, h_dim+h_dim+512)
```

**对齐检查**:
- ✅ `phi_x` 处理输入特征：`(B, N+1, x_dim) -> (B, N+1, h_dim)`
- ✅ `img_embed` 是第一个对象（索引0）
- ✅ FFT处理需要reshape为 `(B, 512, 1)` 格式
- ✅ `obj_embed` 是剩余的对象（索引1:）
- ✅ 拼接顺序：`[obj_embed, img_embed, img_fft]`

### 5. GRU输入维度
**CRASH原始实现** (`CRASH-main/src/Models.py:183`):
```python
self.gru_net = GRUNet(h_dim+h_dim, h_dim, 2, n_layers, dropout=[0.5, 0.0])
# 注意：实际输入是 h_dim+h_dim+512，但GRUNet定义时是 h_dim+h_dim
```

**对齐检查**:
- ⚠️ GRUNet定义时输入维度是 `h_dim+h_dim`，但实际forward时输入是 `h_dim+h_dim+512`
- ✅ 需要检查GRUNet内部是否处理了这个维度差异

### 6. SpatialAttention (OFA) 实现
**CRASH原始实现** (`CRASH-main/src/Models.py:148-158`):
```python
def forward(self, obj_embed, h):
    query1 = self.q1(h[0]).unsqueeze(1) 
    query2 = self.q2(h[1]).unsqueeze(1)
    key = self.wk(obj_embed) 
    value = self.wv(obj_embed)
    attention_score1 = torch.bmm(query1, key.transpose(1,2))/math.sqrt(value.size(-1))
    attention_score2 = torch.bmm(query2, key.transpose(1,2))/math.sqrt(value.size(-1))
    attention_scores = self.alpha1 * attention_score1 + self.alpha2 * attention_score2
    attention_weights = F.softmax(attention_scores, dim=-1)
    weighted_obj_embed = torch.bmm(attention_weights, value)
    return weighted_obj_embed
```

**对齐检查**:
- ✅ 使用双层GRU的隐藏状态作为query
- ✅ 使用alpha1和alpha2加权组合两个query的注意力分数
- ✅ 输出形状：`(B, 1, h_dim)`

### 7. SelfAttAggregate (SAA) 实现
**CRASH原始实现** (`CRASH-main/src/Models.py:86-109`):
```python
def forward(self, hiddens):
    hiddens = hiddens.permute(0,2,1)  # (B, h_dim, T)
    hiddens = self.pos_encoder(hiddens)
    # ... 多头自注意力 ...
    agg_feature = self.dense(output)  # (B, h_dim)
    return agg_feature
```

**对齐检查**:
- ✅ 输入：`(B, h_dim, T)` 或 `(B, T, h_dim)`
- ✅ 使用位置编码
- ✅ 输出：`(B, agg_dim)`，其中 `agg_dim = h_dim`

### 8. 指数损失函数 (_exp_loss)
**CRASH原始实现** (`CRASH-main/src/Models.py:285-293`):
```python
def _exp_loss(self, pred, target, time, toa, fps=10.0):
    target_cls = target[:, 1]
    target_cls = target_cls.to(torch.long)
    penalty = - 0.5 * torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), 
                                 (toa.to(pred.dtype) - time - 1) / fps)
    pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
    neg_loss = self.ce_loss(pred, target_cls)
    loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), 
                                torch.mul(neg_loss, target[:, 0])))
    loss = loss / (self.rho_1 * self.rho_1)
    return loss
```

**对齐检查**:
- ✅ penalty计算：`-0.5 * max(0, (toa - time - 1) / fps)`
- ✅ 正样本损失：`-exp(penalty) * ce_loss`
- ✅ 负样本损失：`ce_loss`
- ✅ 最终损失除以 `rho_1^2`

## 🐛 Debug优先级

1. **首先检查**：与CRASH-main的实现对齐
   - forward方法返回值格式
   - loss计算方式
   - RSD使用方式
   - 特征处理流程

2. **然后检查**：LFCRASH-CBM特有的部分
   - 概念投影层
   - CLIP编码
   - 对齐损失和稀疏损失

3. **最后检查**：数据格式和维度
   - 输入数据形状
   - 中间特征形状
   - 输出形状

## 📝 参考文件

- CRASH原始实现：`/data/sony/Lucas_rename/CRASH-main/CRASH-main/src/Models.py`
- LFCRASH-CBM实现：`/data/sony/LFCRASH/LFCRASH-CBM/src/models_gru.py`
- CRASH评估工具：`/data/sony/LFCRASH/CRASH/src/eval_tools.py`

## ✅ 已验证对齐的部分

- [x] forward方法返回值格式
- [x] loss计算方式（cross_entropy累加）
- [x] RSD使用方式（从t>=2开始）
- [x] 特征处理流程（phi_x -> FFT -> SpatialAttention -> concat）
- [x] SpatialAttention实现（双层GRU query）
- [x] SelfAttAggregate实现（位置编码 + 多头注意力）
- [x] 指数损失函数实现

## ⚠️ 需要注意的差异

1. **GRU输入维度**：CRASH中GRUNet定义时输入是`h_dim+h_dim`，但实际输入是`h_dim+h_dim+512`
2. **rho参数**：CRASH使用`log_rho_1`和`log_rho_2`作为参数，通过property访问`rho_1`和`rho_2`
3. **Variable包装**：CRASH使用`Variable`包装隐藏状态（PyTorch旧版本），新版本可以直接使用tensor
