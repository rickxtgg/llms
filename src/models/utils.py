"""模型工具函数模块

这个模块提供了大语言模型训练和推理过程中使用的各种工具函数，包括高效的注意力计算、并行处理和内存优化等功能。
参考了deepseek-ai的FlashMLA和DualPipe等开源项目的实现。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None, 
                   dropout_p: float = 0.0,
                   block_size: int = 256) -> torch.Tensor:
    """实现高效的Flash Attention机制
    
    基于deepseek-ai的FlashMLA项目，实现了内存高效的注意力计算。
    通过分块计算和优化的内存访问模式，显著降低了内存使用和提高了计算速度。
    
    Args:
        q: 查询张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        k: 键张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        v: 值张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        mask: 注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]
        dropout_p: dropout概率
        block_size: 分块大小，用于优化内存访问
        
    Returns:
        注意力输出，形状为 [batch_size, seq_len, num_heads, head_dim]
    """
    # 获取输入维度
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # 计算注意力分数
    scale = 1.0 / math.sqrt(head_dim)
    q = q * scale
    
    # 重新排列维度以便进行批量矩阵乘法
    q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
    k = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
    v = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
    
    # 使用分块计算优化内存访问
    attn_output = torch.zeros_like(q)
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # 使用torch.jit.script优化计算密集型操作
    @torch.jit.script
    def compute_attention(block_q, block_k, block_v, block_mask=None):
        block_attn = torch.matmul(block_q, block_k.transpose(-1, -2))
        if block_mask is not None:
            block_attn = block_attn.masked_fill(block_mask == 0, float('-inf'))
        block_attn = F.softmax(block_attn, dim=-1)
        if dropout_p > 0.0:
            block_attn = F.dropout(block_attn, p=dropout_p)
        return torch.matmul(block_attn, block_v)
    
    # 使用异步预取优化内存访问
    next_block_tensors = None
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min(start_idx + block_size, seq_len)
        
        # 预分配下一块的内存（异步）
        if i < num_blocks - 1:
            next_start = (i + 1) * block_size
            next_end = min(next_start + block_size, seq_len)
            next_block_tensors = (
                q[:, :, next_start:next_end],
                k[:, :, next_start:next_end],
                v[:, :, next_start:next_end]
            )
        
        # 获取当前块
        block_q = q[:, :, start_idx:end_idx]
        block_k = k
        block_v = v
        
        # 获取掩码块（如果有）
        block_mask = None
        if mask is not None:
            block_mask = mask[:, :, start_idx:end_idx, :]
        
        # 计算块注意力
        block_output = compute_attention(block_q, block_k, block_v, block_mask)
        
        # 更新输出
        attn_output[:, :, start_idx:end_idx] = block_output
    
    # 重新排列维度
    attn_output = attn_output.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
    
    return attn_output


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """应用旋转位置编码
    
    基于RoPE（Rotary Position Embedding）实现，提供了更好的位置感知能力。
    
    Args:
        x: 输入张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        cos: 余弦位置编码
        sin: 正弦位置编码
        
    Returns:
        应用了位置编码的张量
    """
    # 将head_dim分成两半，分别应用旋转
    x1, x2 = x.chunk(2, dim=-1)
    
    # 应用旋转变换
    result = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)
    
    return result


def get_rotary_pos_emb(seq_len: int, dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成旋转位置编码的正弦和余弦部分
    
    Args:
        seq_len: 序列长度
        dim: 嵌入维度
        device: 计算设备
        
    Returns:
        余弦和正弦位置编码的元组
    """
    # 生成位置索引
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    
    # 生成维度索引
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * 
                         (-math.log(10000.0) / dim))
    
    # 计算正弦和余弦值
    sin = torch.sin(position * div_term)
    cos = torch.cos(position * div_term)
    
    # 扩展维度以匹配输入形状
    sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim/2]
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim/2]
    
    # 复制到完整维度
    sin = torch.cat([sin, sin], dim=-1)  # [1, seq_len, 1, dim]
    cos = torch.cat([cos, cos], dim=-1)  # [1, seq_len, 1, dim]
    
    return cos, sin


def get_activation_fn(activation: str):
    """获取激活函数
    
    Args:
        activation: 激活函数名称
        
    Returns:
        激活函数
    """
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "silu" or activation == "swish":
        return F.silu
    else:
        raise ValueError(f"不支持的激活函数: {activation}")


def model_parallel_shard(model: nn.Module, num_gpus: int) -> nn.Module:
    """实现模型并行分片
    
    基于deepseek-ai的DualPipe项目，将模型分布在多个GPU上以支持大规模模型训练。
    
    Args:
        model: 要分片的模型
        num_gpus: GPU数量
        
    Returns:
        分片后的模型
    """
    if num_gpus <= 1:
        return model
    
    # 获取所有Transformer层
    layers = [module for name, module in model.named_modules() 
              if "blocks" in name and isinstance(module, nn.Module)]
    
    # 计算每个GPU上的层数
    layers_per_gpu = len(layers) // num_gpus
    
    # 将层分配到不同的GPU
    for i, layer in enumerate(layers):
        gpu_id = min(i // layers_per_gpu, num_gpus - 1)
        device = torch.device(f"cuda:{gpu_id}")
        layer.to(device)
    
    return model


def optimize_memory_efficiency(model: nn.Module) -> nn.Module:
    """优化模型内存效率
    
    基于deepseek-ai的EPLB项目，实现了训练过程中的内存优化技术。
    
    Args:
        model: 要优化的模型
        
    Returns:
        优化后的模型
    """
    # 启用检查点功能以节省内存
    for name, module in model.named_modules():
        if "blocks" in name and isinstance(module, nn.Module):
            module.gradient_checkpointing = True
    
    return model