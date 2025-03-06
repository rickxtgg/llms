"""Flash Attention模块

这个模块实现了基于deepseek-ai的FlashMLA项目的Flash Attention机制，
提供了内存高效的注意力计算，显著降低了内存使用和提高了计算速度。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union


class FlashAttention(nn.Module):
    """Flash Attention实现
    
    基于deepseek-ai的FlashMLA项目，实现了内存高效的注意力计算。
    通过分块计算和优化的内存访问模式，显著降低了内存使用和提高了计算速度。
    
    参考: https://github.com/deepseek-ai/FlashMLA
    """
    
    def __init__(self, n_embd, n_head, dropout_p=0.1):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.dropout_p = dropout_p
        
        # 投影矩阵
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        
    def forward(self, x, attention_mask=None, layer_past=None):
        """前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, n_embd]
            attention_mask: 注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]
            layer_past: KV缓存，用于加速推理
            
        Returns:
            注意力输出，形状为 [batch_size, seq_len, n_embd]
        """
        batch_size, seq_len, _ = x.size()
        
        # 计算查询、键、值
        q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        
        # 处理KV缓存
        present = None
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)
            present = (k, v)
        
        # 使用utils中的flash_attention函数计算注意力
        from .utils import flash_attention
        attn_output = flash_attention(q, k, v, mask=attention_mask, dropout_p=self.dropout_p)
        
        # 重塑输出并投影
        attn_output = attn_output.reshape(batch_size, seq_len, self.n_embd)
        attn_output = self.out_proj(attn_output)
        
        if present is not None:
            return attn_output, present
        return attn_output


def flash_attention_forward(q, k, v, mask=None, dropout_p=0.0):
    """Flash Attention前向传播函数
    
    这是一个便捷函数，直接调用utils.flash_attention实现。
    
    Args:
        q: 查询张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        k: 键张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        v: 值张量，形状为 [batch_size, seq_len, num_heads, head_dim]
        mask: 注意力掩码，形状为 [batch_size, 1, seq_len, seq_len]
        dropout_p: dropout概率
        
    Returns:
        注意力输出，形状为 [batch_size, seq_len, num_heads, head_dim]
    """
    from .utils import flash_attention
    return flash_attention(q, k, v, mask, dropout_p)