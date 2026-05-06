#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练结果，检查Optuna优化和完整训练的差异
"""

import os
import re
import json
import numpy as np
from pathlib import Path

def extract_epoch_results(log_file):
    """从日志文件中提取每个epoch的评估结果"""
    results = []
    
    if not os.path.exists(log_file):
        return results
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 提取epoch评估结果
    pattern = r'Epoch \[(\d+)/\d+\].*?AP \(video-level\): ([\d.]+).*?mTTA: ([\d.]+).*?TTA@R80: ([\d.]+).*?P@R80: ([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        epoch = int(match[0])
        ap = float(match[1])
        mtta = float(match[2])
        tta_r80 = float(match[3])
        p_r80 = float(match[4])
        results.append({
            'epoch': epoch,
            'ap': ap,
            'mtta': mtta,
            'tta_r80': tta_r80,
            'p_r80': p_r80
        })
    
    return results

def extract_loss_trends(log_file):
    """提取训练损失趋势"""
    losses = []
    
    if not os.path.exists(log_file):
        return losses
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 匹配 "Epoch [X/Y] 完成: Total=loss"
            match = re.search(r'Epoch \[(\d+)/\d+\].*?完成.*?Total=([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                losses.append({'epoch': epoch, 'loss': loss})
    
    return losses

def analyze_dataset(dataset_name, timestamp):
    """分析单个数据集的结果"""
    log_file = f"logs/full_training_{dataset_name}_{timestamp}.log"
    result_file = f"output/full_training_{timestamp}/best_{dataset_name}/results.json"
    
    print(f"\n{'='*80}")
    print(f"数据集: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # 提取epoch结果
    epoch_results = extract_epoch_results(log_file)
    loss_trends = extract_loss_trends(log_file)
    
    if epoch_results:
        print(f"\n📊 Epoch评估结果:")
        print(f"{'Epoch':<8} {'AP':<10} {'mTTA':<10} {'TTA@R80':<12} {'P@R80':<10}")
        print("-" * 60)
        for r in epoch_results[:10]:  # 只显示前10个
            print(f"{r['epoch']:<8} {r['ap']:<10.4f} {r['mtta']:<10.4f} {r['tta_r80']:<12.4f} {r['p_r80']:<10.4f}")
        if len(epoch_results) > 10:
            print(f"... (共{len(epoch_results)}个epoch评估结果)")
        
        # 分析最佳epoch
        best_result = max(epoch_results, key=lambda x: x['ap'])
        print(f"\n✅ 最佳结果: Epoch {best_result['epoch']}, AP={best_result['ap']:.4f}")
        
        # 检查是否有提升
        if len(epoch_results) > 1:
            first_ap = epoch_results[0]['ap']
            last_ap = epoch_results[-1]['ap']
            best_ap = best_result['ap']
            print(f"   第1个评估epoch AP: {first_ap:.4f}")
            print(f"   最后1个评估epoch AP: {last_ap:.4f}")
            print(f"   最佳AP: {best_ap:.4f}")
            if best_result['epoch'] == epoch_results[0]['epoch']:
                print(f"   ⚠️  警告: 最佳epoch是第一个评估epoch，之后没有提升！")
    else:
        print("   ❌ 未找到epoch评估结果")
    
    if loss_trends:
        print(f"\n📉 训练损失趋势 (前10个epoch):")
        print(f"{'Epoch':<8} {'Loss':<10}")
        print("-" * 20)
        for l in loss_trends[:10]:
            print(f"{l['epoch']:<8} {l['loss']:<10.4f}")
        if len(loss_trends) > 10:
            print(f"... (共{len(loss_trends)}个epoch损失记录)")
    
    # 读取最终结果
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            final_result = json.load(f)
        print(f"\n📋 最终保存的结果:")
        print(f"   AP: {final_result['ap']:.4f}")
        print(f"   mTTA: {final_result['mtta']:.4f}")
        print(f"   TTA@R80: {final_result['tta_r80']:.4f}")
        print(f"   P@R80: {final_result['p_r80']:.4f}")
        print(f"   最佳Epoch: {final_result['best_epoch']}")
    
    return epoch_results, loss_trends

def compare_with_optuna():
    """对比Optuna优化时的结果"""
    print(f"\n{'='*80}")
    print("对比Optuna优化结果")
    print(f"{'='*80}")
    
    # 查找Optuna日志
    optuna_logs = []
    if os.path.exists("logs"):
        for f in os.listdir("logs"):
            if "optuna" in f.lower() and f.endswith(".log"):
                optuna_logs.append(f"logs/{f}")
    
    if optuna_logs:
        print(f"\n找到{len(optuna_logs)}个Optuna日志文件")
        for log_file in optuna_logs[:3]:  # 只显示前3个
            print(f"  - {log_file}")
    else:
        print("\n未找到Optuna日志文件")

def main():
    timestamp = "20260114_121308"
    
    print("="*80)
    print("训练结果分析报告")
    print("="*80)
    
    datasets = ['dad', 'crash', 'a3d']
    all_results = {}
    
    for dataset in datasets:
        epoch_results, loss_trends = analyze_dataset(dataset, timestamp)
        all_results[dataset] = {
            'epoch_results': epoch_results,
            'loss_trends': loss_trends
        }
    
    # 对比分析
    compare_with_optuna()
    
    # 总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    
    for dataset in datasets:
        epoch_results = all_results[dataset]['epoch_results']
        if epoch_results:
            best_result = max(epoch_results, key=lambda x: x['ap'])
            print(f"\n{dataset.upper()}:")
            print(f"  最佳AP: {best_result['ap']:.4f} (Epoch {best_result['epoch']})")
            if best_result['epoch'] <= 5:
                print(f"  ⚠️  最佳epoch过早 (<=5)，可能存在过拟合或评估问题")

if __name__ == "__main__":
    main()
