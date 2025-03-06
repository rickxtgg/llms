"""评估指标模块

这个模块提供了评估语言模型性能的各种指标，包括困惑度、准确率等。
"""

import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_perplexity(loss: float) -> float:
    """计算困惑度
    
    Args:
        loss: 交叉熵损失
        
    Returns:
        困惑度值
    """
    return math.exp(loss)


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """计算预测准确率
    
    Args:
        predictions: 模型预测结果，形状为 [batch_size, seq_len, vocab_size] 或 [batch_size * seq_len, vocab_size]
        targets: 目标标签，形状为 [batch_size, seq_len] 或 [batch_size * seq_len]
        ignore_index: 忽略的标签索引，默认为-100
        
    Returns:
        准确率（0-1之间）
    """
    if predictions.dim() > 2:
        # 如果是3D张量，转换为2D
        predictions = predictions.reshape(-1, predictions.size(-1))
        targets = targets.reshape(-1)
    
    # 获取每个位置的最大概率对应的索引作为预测结果
    pred_indices = torch.argmax(predictions, dim=-1)
    
    # 创建掩码，排除忽略的索引
    mask = (targets != ignore_index)
    
    # 计算准确率
    correct = (pred_indices == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def calculate_token_level_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                                 ignore_index: int = -100) -> Dict[str, float]:
    """计算token级别的评估指标
    
    Args:
        predictions: 模型预测结果，形状为 [batch_size, seq_len, vocab_size] 或 [batch_size * seq_len, vocab_size]
        targets: 目标标签，形状为 [batch_size, seq_len] 或 [batch_size * seq_len]
        ignore_index: 忽略的标签索引，默认为-100
        
    Returns:
        包含各种指标的字典
    """
    if predictions.dim() > 2:
        # 如果是3D张量，转换为2D
        predictions = predictions.reshape(-1, predictions.size(-1))
        targets = targets.reshape(-1)
    
    # 获取每个位置的最大概率对应的索引作为预测结果
    pred_indices = torch.argmax(predictions, dim=-1)
    
    # 创建掩码，排除忽略的索引
    mask = (targets != ignore_index)
    valid_preds = pred_indices[mask]
    valid_targets = targets[mask]
    
    # 计算准确率
    accuracy = (valid_preds == valid_targets).float().mean().item()
    
    # 计算top-k准确率（top-5）
    _, top5_indices = torch.topk(predictions[mask], k=5, dim=-1)
    top5_accuracy = torch.any(top5_indices == valid_targets.unsqueeze(-1), dim=-1).float().mean().item()
    
    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy
    }


def calculate_sequence_level_metrics(generated_texts: List[str], reference_texts: List[str]) -> Dict[str, float]:
    """计算序列级别的评估指标
    
    Args:
        generated_texts: 模型生成的文本列表
        reference_texts: 参考文本列表
        
    Returns:
        包含各种指标的字典
    """
    # 这里可以添加更多序列级别的指标，如BLEU、ROUGE等
    # 这需要额外的依赖，如nltk、rouge_score等
    
    # 简单的长度比率指标
    length_ratios = [len(gen) / max(1, len(ref)) for gen, ref in zip(generated_texts, reference_texts)]
    avg_length_ratio = sum(length_ratios) / len(length_ratios)
    
    return {
        "avg_length_ratio": avg_length_ratio
    }


def evaluate_model_outputs(model_outputs: Dict[str, torch.Tensor], 
                          targets: torch.Tensor) -> Dict[str, float]:
    """评估模型输出的综合指标
    
    Args:
        model_outputs: 模型输出的字典，包含logits等
        targets: 目标标签
        
    Returns:
        包含各种评估指标的字典
    """
    logits = model_outputs.get("logits", model_outputs.get("predictions"))
    loss = model_outputs.get("loss")
    
    metrics = {}
    
    # 如果有损失，计算困惑度
    if loss is not None:
        metrics["perplexity"] = calculate_perplexity(loss.item() if isinstance(loss, torch.Tensor) else loss)
    
    # 如果有logits，计算准确率
    if logits is not None:
        token_metrics = calculate_token_level_metrics(logits, targets)
        metrics.update(token_metrics)
    
    return metrics


def print_evaluation_results(metrics: Dict[str, float]):
    """打印评估结果
    
    Args:
        metrics: 包含各种评估指标的字典
    """
    print("\n===== 评估结果 =====")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("=====================\n")


if __name__ == "__main__":
    # 简单测试
    import torch.nn.functional as F
    
    # 模拟一些数据
    batch_size, seq_len, vocab_size = 2, 5, 10
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 计算损失
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    
    # 计算指标
    perplexity = calculate_perplexity(loss.item())
    accuracy = calculate_accuracy(logits, targets)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Accuracy: {accuracy:.4f}")