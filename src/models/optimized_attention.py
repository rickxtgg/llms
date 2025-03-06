"""优化的注意力机制模块

这个模块整合了deepseek-ai的多个开源项目的优化技术，包括：
1. FlashMLA - 高效的注意力计算
2. DualPipe - 模型并行处理
3. EPLB - 内存优化

提供了一个统一的接口，使用户能够轻松地在项目中应用这些优化技术。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union

from .utils import flash_attention, apply_rotary_pos_emb, get_rotary_pos_emb
from .utils import model_parallel_shard, optimize_memory_efficiency
from .flash_mla import BlockSparseAttention, MultiQueryAttention, KVCache


class OptimizedAttention(nn.Module):
    """优化的注意力机制
    
    整合了deepseek-ai的多个开源项目的优化技术，提供了一个统一的接口。
    支持以下优化技术：
    1. Flash Attention - 高效的注意力计算
    2. 块稀疏注意力 - 降低计算复杂度
    3. 多查询注意力 - 减少内存使用
    4. 旋转位置编码 - 提供更好的位置感知能力
    5. KV缓存 - 加速自回归生成
    6. 模型并行 - 支持大规模模型训练
    7. 梯度检查点 - 优化内存使用
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        
        # 获取注意力优化配置
        attn_config = getattr(config, 'attention_optimization', {})
        self.use_flash_attention = attn_config.get('flash_attention', False)
        self.use_block_sparse = attn_config.get('block_sparse', False)
        self.use_multi_query = attn_config.get('multi_query', False)
        self.use_rotary_pos_emb = attn_config.get('rotary_position_embeddings', False)
        self.use_kv_cache = attn_config.get('kv_cache', False)
        
        # 获取内存优化配置
        memory_config = getattr(config, 'memory_optimization', {})
        self.use_gradient_checkpointing = memory_config.get('gradient_checkpointing', False)
        
        # 获取模型并行配置
        parallel_config = getattr(config, 'model_parallel', {})
        self.use_model_parallel = parallel_config.get('enabled', False)
        self.num_gpus = parallel_config.get('num_gpus', 1) if self.use_model_parallel else 1
        
        # 根据配置选择注意力实现
        if self.use_block_sparse:
            self.attn = BlockSparseAttention(
                n_embd=self.n_embd,
                n_head=self.n_head,
                block_size=attn_config.get('block_size', 32),
                sparsity=attn_config.get('sparsity', 0.8)
            )
        elif self.use_multi_query:
            self.attn = MultiQueryAttention(
                n_embd=self.n_embd,
                n_head=self.n_head,
                head_dim=self.head_dim
            )
        else:
            # 使用标准多头注意力
            from .attention import MultiHeadAttention
            self.attn = MultiHeadAttention(config)
        
        # 层归一化和残差连接
        self.ln = nn.LayerNorm(self.n_embd)
        
        # 初始化KV缓存（如果启用）
        if self.use_kv_cache:
            self.kv_cache = None  # 延迟初始化
    
    def init_kv_cache(self, batch_size, max_seq_len, n_layers):
        """初始化KV缓存
        
        Args:
            batch_size: 批次大小
            max_seq_len: 最大序列长度
            n_layers: 层数
        """
        if self.use_kv_cache and self.kv_cache is None:
            self.kv_cache = KVCache(
                max_batch_size=batch_size,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                n_heads=self.n_head,
                head_dim=self.head_dim
            )
    
    def forward(self, x, layer_idx=0, attention_mask=None, layer_past=None):
        """前向传播
        
        Args:
            x: 输入张量
            layer_idx: 层索引（用于KV缓存和模型并行）
            attention_mask: 注意力掩码
            layer_past: KV缓存的过去状态
            
        Returns:
            注意力输出，如果使用KV缓存，则返回(输出, 当前状态)
        """
        # 获取归一化输入
        norm_x = self.ln(x)
        
        # 获取KV缓存（如果启用）
        if layer_past is None and self.use_kv_cache and self.kv_cache is not None:
            layer_past = self.kv_cache.get(layer_idx)
        
        # 使用梯度检查点以节省内存（如果启用）
        if self.use_gradient_checkpointing and self.training and layer_past is None:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # 修复：确保在梯度检查点中正确处理attention_mask
            if attention_mask is not None:
                attn_output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.attn), 
                    norm_x, attention_mask=attention_mask
                )
            else:
                attn_output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.attn), 
                    norm_x
                )
                
            # 处理返回值
            if isinstance(attn_output, tuple):
                output, present = attn_output
                return x + output, present
            else:
                return x + attn_output
        else:
            # 应用旋转位置编码（如果启用）
            if self.use_rotary_pos_emb:
                B, T, C = norm_x.size()
                
                # 处理不同类型的注意力实现
                if hasattr(self.attn, 'query') and hasattr(self.attn, 'key'):
                    # 标准多头注意力或块稀疏注意力
                    q = self.attn.query(norm_x).view(B, T, self.n_head, self.head_dim)
                    k = self.attn.key(norm_x).view(B, T, self.n_head, self.head_dim)
                    v = self.attn.value(norm_x).view(B, T, self.n_head, self.head_dim)
                    
                    # 应用旋转位置编码
                    from .utils import get_rotary_pos_emb, apply_rotary_pos_emb
                    cos, sin = get_rotary_pos_emb(T, self.head_dim, norm_x.device)
                    q = apply_rotary_pos_emb(q, cos, sin)
                    k = apply_rotary_pos_emb(k, cos, sin)
                    
                    # 处理KV缓存
                    present = None
                    if layer_past is not None:
                        past_k, past_v = layer_past
                        k = torch.cat((past_k, k), dim=1)
                        v = torch.cat((past_v, v), dim=1)
                        present = (k, v)
                    
                    # 使用flash_attention直接计算
                    from .utils import flash_attention
                    
                    # 创建注意力掩码
                    mask = None
                    if layer_past is None:  # 训练模式
                        # 创建因果掩码
                        mask = torch.tril(torch.ones(T, k.size(1), device=norm_x.device)).view(1, 1, T, k.size(1))
                    
                    # 如果提供了外部掩码，与因果掩码结合
                    if attention_mask is not None:
                        if mask is not None:
                            mask = mask & attention_mask
                        else:
                            mask = attention_mask
                    
                    # 计算注意力输出
                    attn_output = flash_attention(q, k, v, mask)
                    attn_output = attn_output.reshape(B, T, C)
                    attn_output = self.attn.out_proj(attn_output) if hasattr(self.attn, 'out_proj') else attn_output
                    
                    if present is not None:
                        # 更新KV缓存
                        if self.use_kv_cache and self.kv_cache is not None:
                            self.kv_cache.update(layer_idx, *present)
                        return x + attn_output, present
                    return x + attn_output
                elif hasattr(self.attn, 'q_proj') and hasattr(self.attn, 'k_proj'):
                    # 多查询注意力的旋转位置编码已在MultiQueryAttention内部处理
                    pass
            
            # 正常前向传播
            if layer_past is not None:
                if hasattr(self.attn, 'layer_past'):
                    self.attn.layer_past = layer_past
                
                attn_output = self.attn(norm_x, attention_mask=attention_mask, layer_past=layer_past)
                
                # 处理返回值
                if isinstance(attn_output, tuple):
                    output, present = attn_output
                    # 更新KV缓存
                    if self.use_kv_cache and self.kv_cache is not None and present is not None:
                        self.kv_cache.update(layer_idx, *present)
                    return x + output, present
                else:
                    return x + attn_output
            else:
                attn_output = self.attn(norm_x, attention_mask=attention_mask)
                
                # 处理返回值
                if isinstance(attn_output, tuple):
                    output, present = attn_output
                    return x + output, present
                else:
                    return x + attn_output


class OptimizedTransformerBlock(nn.Module):
    """优化的Transformer块
    
    整合了deepseek-ai的多个开源项目的优化技术，包括注意力机制优化、前馈网络优化等。
    """
    
    def __init__(self, config):
        super().__init__()
        self.attn = OptimizedAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
        
        # 是否使用梯度检查点（用于内存优化）
        memory_config = getattr(config, 'memory_optimization', {})
        self.gradient_checkpointing = memory_config.get('gradient_checkpointing', False)
    
    def forward(self, x, layer_idx=0, attention_mask=None, layer_past=None):
        """前向传播
        
        Args:
            x: 输入张量
            layer_idx: 层索引
            attention_mask: 注意力掩码
            layer_past: KV缓存的过去状态
            
        Returns:
            Transformer块的输出
        """
        # 注意力层
        attn_output = self.attn(x, layer_idx=layer_idx, attention_mask=attention_mask, layer_past=layer_past)
        
        # 处理KV缓存
        present = None
        if isinstance(attn_output, tuple):
            x, present = attn_output
        else:
            x = attn_output
        
        # 前馈网络
        if self.gradient_checkpointing and self.training and present is None:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            x = x + torch.utils.checkpoint.checkpoint(create_custom_forward(self.mlp), self.ln_2(x))
        else:
            x = x + self.mlp(self.ln_2(x))
        
        if present is not None:
            return x, present
        return x


def apply_optimizations(model, config):
    """应用deepseek-ai的优化技术到模型
    
    Args:
        model: 要优化的模型
        config: 配置对象
        
    Returns:
        优化后的模型
    """
    # 应用内存优化
    memory_config = getattr(config, 'memory_optimization', {})
    if memory_config.get('enabled', False):
        model = optimize_memory_efficiency(model)
    
    # 应用模型并行
    parallel_config = getattr(config, 'model_parallel', {})
    if parallel_config.get('enabled', False):
        num_gpus = parallel_config.get('num_gpus', 1)
        model = model_parallel_shard(model, num_gpus)
    
    return model