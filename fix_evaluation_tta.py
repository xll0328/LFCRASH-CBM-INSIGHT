#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复evaluation函数中的TTA计算问题，添加调试输出
"""

import sys
import os
import numpy as np

# 添加CRASH路径
crash_path = os.path.join(os.path.dirname(__file__), '..', 'CRASH')
sys.path.insert(0, crash_path)

from src.eval_tools import evaluation

def evaluation_with_debug(all_pred, all_labels, time_of_accidents, fps=20.0):
    """
    带调试输出的evaluation函数
    """
    print(f"\n{'='*60}")
    print(f"evaluation函数调试信息")
    print(f"{'='*60}")
    print(f"输入数据:")
    print(f"  all_pred shape: {all_pred.shape}")
    print(f"  all_pred range: [{all_pred.min():.4f}, {all_pred.max():.4f}]")
    print(f"  all_pred mean: {all_pred.mean():.4f}")
    print(f"  all_labels: {all_labels}")
    print(f"  all_labels sum: {np.sum(all_labels)}")
    print(f"  time_of_accidents: {time_of_accidents[:5]}... (showing first 5)")
    print(f"  fps: {fps}")
    
    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]
        else:
            pred = all_pred[idx, :]
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps
    
    print(f"\n处理后的数据:")
    print(f"  n_frames: {n_frames}")
    print(f"  min_pred: {min_pred:.4f}")
    print(f"  total_seconds: {total_seconds:.2f}")
    
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        if cnt >= n_frames:
            break
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0 
        for i in range(len(preds_eval)):
            tp = np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0: 
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    
    print(f"\n计算统计:")
    print(f"  有效阈值数量 (cnt): {cnt}")
    print(f"  Precision非零数量: {np.count_nonzero(Precision)}")
    print(f"  Recall非零数量: {np.count_nonzero(Recall)}")
    print(f"  Time非零数量: {np.count_nonzero(Time)}")
    
    if cnt == 0:
        print(f"\n⚠️  警告: 没有有效的阈值，返回默认值")
        return 0.0, 5.0, 5.0, 0.0
    
    new_index = np.argsort(Recall[:cnt])
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    
    print(f"\n去重后:")
    print(f"  rep_index长度: {len(rep_index)}")
    print(f"  rep_index: {rep_index[:10]}... (showing first 10)")
    
    if len(rep_index) == 0:
        print(f"\n⚠️  警告: rep_index为空，返回默认值")
        return 0.0, 5.0, 5.0, 0.0
    
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    
    print(f"\n最终指标:")
    print(f"  new_Recall范围: [{new_Recall.min():.4f}, {new_Recall.max():.4f}]")
    print(f"  new_Time范围: [{new_Time.min():.4f}, {new_Time.max():.4f}]")
    print(f"  new_Precision范围: [{new_Precision.min():.4f}, {new_Precision.max():.4f}]")
    
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    mTTA = np.mean(new_Time) * total_seconds
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    a = np.where(new_Recall>=0.8)
    if len(a[0]) > 0:
        P_R80 = new_Precision[a[0][0]]
    else:
        P_R80 = 0.0
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    
    print(f"\n计算结果:")
    print(f"  AP: {AP:.4f}")
    print(f"  mTTA: {mTTA:.4f}")
    print(f"  TTA@R80: {TTA_R80:.4f}")
    print(f"  P@R80: {P_R80:.4f}")
    
    return AP, mTTA, TTA_R80, P_R80

if __name__ == "__main__":
    # 测试
    n_videos = 10
    n_frames = 50
    
    # 测试1: 正常情况
    print("\n" + "="*60)
    print("测试1: 正常情况")
    print("="*60)
    all_pred = np.random.rand(n_videos, n_frames) * 0.5 + 0.5
    all_labels = np.random.randint(0, 2, n_videos)
    all_toa = np.random.randint(10, n_frames-10, n_videos).astype(float)
    evaluation_with_debug(all_pred, all_labels, all_toa, fps=10.0)
    
    # 测试2: 所有预测值相同
    print("\n" + "="*60)
    print("测试2: 所有预测值相同")
    print("="*60)
    all_pred = np.ones((n_videos, n_frames)) * 0.5
    all_labels = np.random.randint(0, 2, n_videos)
    all_toa = np.random.randint(10, n_frames-10, n_videos).astype(float)
    evaluation_with_debug(all_pred, all_labels, all_toa, fps=10.0)
    
    # 测试3: 所有标签都是0
    print("\n" + "="*60)
    print("测试3: 所有标签都是0")
    print("="*60)
    all_pred = np.random.rand(n_videos, n_frames)
    all_labels = np.zeros(n_videos)
    all_toa = np.random.randint(10, n_frames-10, n_videos).astype(float)
    evaluation_with_debug(all_pred, all_labels, all_toa, fps=10.0)
