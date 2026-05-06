# TTA计算问题修复说明

## 问题根源

**问题**: mTTA和TTA@R80都是5.0，这不符合预期。

**根本原因**: 
1. 我们使用了`np.tile(all_probs[:, None], (1, n_frames))`将所有帧的预测值设为相同
2. 当所有帧的预测值都相同时，evaluation函数中的TTA计算会出现问题：
   - 当阈值Th低于预测值时，所有帧都会被检测到，`tp[0][0]`总是0（第一帧）
   - 这导致`time += tp[0][0] / float(time_of_accidents[i])` = 0 / toa = 0
   - 所以`Time[cnt] = (1-time/counter) = (1-0) = 1.0`
   - 最终`mTTA = np.mean(new_Time) * total_seconds = 1.0 * 5.0 = 5.0`

## CRASH原始代码的期望

CRASH的`evaluation`函数期望接收的是**frame-level predictions**，即：
- `all_pred`的形状应该是`(n_videos, n_frames)`
- 每一帧都应该有不同的预测值
- 这样才能正确计算TTA（Time-to-Accident）

## 修复方案

### 修复前（错误）:
```python
# 只使用最后一帧的输出
last_output = all_outputs[-1]  # (B, 2)
video_probs = torch.softmax(last_output, dim=-1)[:, 1]  # (B,)
all_probs.append(video_probs.cpu().numpy())

# 然后用np.tile复制到所有帧
all_pred = np.tile(all_probs[:, None], (1, n_frames))  # 所有帧都相同！
```

### 修复后（正确）:
```python
# 使用all_outputs中每一帧的输出
frame_probs_list = []
for frame_idx in range(min(len(all_outputs), n_frames)):
    frame_output = all_outputs[frame_idx]  # (B, 2)
    frame_probs = torch.softmax(frame_output, dim=-1)[:, 1]  # (B,) - 正类概率
    frame_probs_list.append(frame_probs.cpu().numpy())

# 转换为(batch_size, n_frames)
frame_probs_array = np.array(frame_probs_list).T  # (batch_size, n_frames)
all_pred_frame.append(frame_probs_array)

# 最后concatenate得到(n_videos, n_frames)
all_pred = np.concatenate(all_pred_frame, axis=0)  # 每帧都有不同的预测值！
```

## 修复位置

1. **评估循环**（第385-498行）：修复了每个epoch评估时的frame-level predictions构造
2. **最终测试循环**（第548-664行）：修复了最终测试时的frame-level predictions构造

## 验证

修复后，`all_pred`的形状仍然是`(n_videos, n_frames)`，但现在每一帧都有不同的预测值，这样evaluation函数才能正确计算TTA。

## 下一步

需要重新运行训练，验证修复后的TTA值是否正常。
