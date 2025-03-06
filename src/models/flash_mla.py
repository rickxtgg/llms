"""FlashMLA模块

这个模块实现了基于deepseek-ai的FlashMLA项目的高效注意力机制，
包括块稀疏注意力、多查询注意力和高效的KV缓存管理等优化技术。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union


class BlockSparseAttention(nn.Module):
    """块稀疏注意力机制
    
    基于deepseek-ai的FlashMLA项目，实现了高效的块稀疏注意力计算。
    通过只计算重要的注意力块，显著降低了计算复杂度和内存使用。
    
    参考: https://github.com/deepseek-ai/FlashMLA
    """
    
    def __init__(self, n_embd, n_head, block_size=32, sparsity=0.8):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.sparsity = sparsity
        
        # 投影矩阵
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        
        # 块稀疏掩码生成器
        self.mask_generator = BlockSparseMaskGenerator(block_size, sparsity)
        
    def forward(self, x, attention_mask=None, layer_past=None):
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
            
        # 重新排列维度
        q = q.permute(0, 2, 1, 3)  # [batch_size, n_head, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)  # [batch_size, n_head, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [batch_size, n_head, seq_len, head_dim]
        
        # 生成块稀疏掩码
        sparse_mask = self.mask_generator(seq_len, k.size(2), device=x.device)
        
        # 创建因果掩码（如果在训练模式）
        if layer_past is None:
            causal_mask = torch.tril(torch.ones(seq_len, k.size(2), device=x.device)).view(1, 1, seq_len, k.size(2))
            sparse_mask = sparse_mask & causal_mask
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            sparse_mask = sparse_mask & attention_mask
        
        # 计算块稀疏注意力
        attn_output = self._block_sparse_attention(q, k, v, sparse_mask)
        
        # 重新排列维度并投影
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.n_embd)
        attn_output = self.out_proj(attn_output)
        
        if present is not None:
            return attn_output, present
        return attn_output
    
    def _block_sparse_attention(self, q, k, v, sparse_mask):
        batch_size, n_head, seq_len, head_dim = q.size()
        
        # 缩放查询并初始化输出
        q = q * (1.0 / math.sqrt(head_dim))
        attn_output = torch.zeros_like(q)
        
        # 计算块的数量
        n_blocks_rows = (seq_len + self.block_size - 1) // self.block_size
        n_blocks_cols = (k.size(2) + self.block_size - 1) // self.block_size
        
        # 获取活跃块的索引
        active_blocks = torch.nonzero(sparse_mask)
        
        # 优化：将活跃块分组以提高并行效率
        block_group_size = 16  # 增加每组处理的块数以提高并行度
        n_groups = (len(active_blocks) + block_group_size - 1) // block_group_size
        
        # 使用torch.jit.script优化计算密集型操作
        @torch.jit.script
        def compute_block_attention(group_q, group_k, group_v):
            block_attn = torch.matmul(group_q, group_k.transpose(-1, -2))
            block_attn = F.softmax(block_attn, dim=-1)
            return torch.matmul(block_attn, group_v)
        
        # 使用异步预取优化内存访问
        next_group_tensors = None
        for group_idx in range(n_groups):
            # 处理当前组
            start_idx = group_idx * block_group_size
            end_idx = min(start_idx + block_group_size, len(active_blocks))
            group_blocks = active_blocks[start_idx:end_idx]
            
            # 预分配下一组的内存（异步）
            if group_idx < n_groups - 1:
                next_start_idx = (group_idx + 1) * block_group_size
                next_end_idx = min(next_start_idx + block_group_size, len(active_blocks))
                next_group_blocks = active_blocks[next_start_idx:next_end_idx]
                next_max_block_rows = max(min(self.block_size, seq_len - i * self.block_size)
                                       for i, _ in next_group_blocks)
                next_max_block_cols = max(min(self.block_size, k.size(2) - j * self.block_size)
                                       for _, j in next_group_blocks)
                next_group_tensors = (
                    torch.zeros((batch_size, n_head, len(next_group_blocks), next_max_block_rows, head_dim),
                              dtype=q.dtype, device=q.device),
                    torch.zeros((batch_size, n_head, len(next_group_blocks), next_max_block_cols, head_dim),
                              dtype=k.dtype, device=k.device),
                    torch.zeros((batch_size, n_head, len(next_group_blocks), next_max_block_cols, head_dim),
                              dtype=v.dtype, device=v.device)
                )
            
            # 使用预分配的内存或创建新的内存
            if next_group_tensors is not None:
                group_q, group_k, group_v = next_group_tensors
            else:
                group_q = torch.zeros((batch_size, n_head, len(group_blocks), max_block_rows, head_dim),
                                    dtype=q.dtype, device=q.device)
                group_k = torch.zeros((batch_size, n_head, len(group_blocks), max_block_cols, head_dim),
                                    dtype=k.dtype, device=k.device)
                group_v = torch.zeros((batch_size, n_head, len(group_blocks), max_block_cols, head_dim),
                                    dtype=v.dtype, device=v.device)
            
            # 使用异步内存拷贝
            for block_idx, (i, j) in enumerate(group_blocks):
                row_start = i * self.block_size
                row_end = min(row_start + self.block_size, seq_len)
                col_start = j * self.block_size
                col_end = min(col_start + self.block_size, k.size(2))
                
                # 使用异步内存拷贝
                with torch.cuda.stream(torch.cuda.Stream()):
                    group_q[:, :, block_idx, :(row_end-row_start)].copy_(q[:, :, row_start:row_end])
                    group_k[:, :, block_idx, :(col_end-col_start)].copy_(k[:, :, col_start:col_end])
                    group_v[:, :, block_idx, :(col_end-col_start)].copy_(v[:, :, col_start:col_end])
            
            # 使用优化后的注意力计算
            block_output = compute_block_attention(group_q, group_k, group_v)
            
            # 更新输出张量
            for block_idx, (i, j) in enumerate(group_blocks):
                row_start = i * self.block_size
                row_end = min(row_start + self.block_size, seq_len)
                attn_output[:, :, row_start:row_end] += block_output[:, :, block_idx, :(row_end-row_start)]
        
        return attn_output


class BlockSparseMaskGenerator:
    """块稀疏掩码生成器
    
    生成用于块稀疏注意力的掩码。
    """
    
    def __init__(self, block_size, sparsity):
        self.block_size = block_size
        self.sparsity = sparsity
    
    def __call__(self, rows, cols, device):
        """生成块稀疏掩码
        
        Args:
            rows: 行数
            cols: 列数
            device: 计算设备
            
        Returns:
            块稀疏掩码，形状为 [n_blocks_rows, n_blocks_cols]
        """
        # 计算块的数量
        n_blocks_rows = (rows + self.block_size - 1) // self.block_size
        n_blocks_cols = (cols + self.block_size - 1) // self.block_size
        
        # 创建全零掩码
        mask = torch.zeros(n_blocks_rows, n_blocks_cols, dtype=torch.bool, device=device)
        
        # 对角线块始终保留（用于自回归属性）
        for i in range(min(n_blocks_rows, n_blocks_cols)):
            mask[i, :i+1] = True
        
        # 随机选择一些非对角线块（基于稀疏度）
        if self.sparsity < 1.0:
            n_active = int((1.0 - self.sparsity) * n_blocks_rows * n_blocks_cols)
            indices = torch.randperm(n_blocks_rows * n_blocks_cols, device=device)[:n_active]
            rows_idx = indices // n_blocks_cols
            cols_idx = indices % n_blocks_cols
            mask[rows_idx, cols_idx] = True
        
        return mask


class MultiQueryAttention(nn.Module):
    """多查询注意力机制
    
    基于deepseek-ai的FlashMLA项目，实现了内存高效的多查询注意力。
    通过为所有注意力头共享相同的键和值，显著减少了内存使用。
    
    参考: https://github.com/deepseek-ai/FlashMLA
    """
    
    def __init__(self, n_embd, n_head, head_dim=None):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        # 如果未指定head_dim，则从n_embd计算
        self.head_dim = head_dim if head_dim is not None else n_embd // n_head
        
        # 多头查询投影
        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim)
        # 单头键值投影（共享给所有注意力头）
        self.k_proj = nn.Linear(n_embd, self.head_dim)
        self.v_proj = nn.Linear(n_embd, self.head_dim)
        
        self.out_proj = nn.Linear(n_head * self.head_dim, n_embd)
        
    def forward(self, x, attention_mask=None, layer_past=None):
        batch_size, seq_len, _ = x.size()
        
        # 计算查询、键、值
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim)
        
        # 处理KV缓存
        present = None
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)
            present = (k, v)
        
        # 扩展键和值到所有头
        k = k.expand(-1, -1, self.n_head, -1)
        v = v.expand(-1, -1, self.n_head, -1)
        
        # 重新排列维度
        q = q.permute(0, 2, 1, 3)  # [batch_size, n_head, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)  # [batch_size, n_head, seq_len, head_dim]
        v = v.permute(0, 2, 1, 3)  # [batch_size, n_head, seq_len, head_dim]
        
        # 计算注意力分数并应用缩放
        q = q * (1.0 / math.sqrt(self.head_dim))
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        
        # 创建注意力掩码
        mask = None
        if layer_past is None:  # 训练模式
            # 创建因果掩码，并确保维度正确
            mask = torch.tril(torch.ones(seq_len, k.size(2), device=x.device))
            # 添加批次和头部维度
            mask = mask.view(1, 1, seq_len, k.size(2))
            # 确保掩码类型与注意力权重一致
            mask = mask.to(dtype=attn_weights.dtype)
        
        # 如果提供了外部掩码，与因果掩码结合
        if attention_mask is not None:
            # 确保外部掩码维度正确
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # 将掩码转换为float类型以支持加法
            attention_mask = attention_mask.to(dtype=attn_weights.dtype)
            # 使用加法而不是与运算来组合掩码
            if mask is not None:
                mask = mask + attention_mask
            else:
                mask = attention_mask
        
        # 应用掩码
        if mask is not None:
            # 使用masked_fill优化性能
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        # 使用torch.jit.script优化softmax计算
        @torch.jit.script
        def apply_softmax(x: torch.Tensor) -> torch.Tensor:
            return F.softmax(x, dim=-1)
        
        attn_weights = apply_softmax(attn_weights)
        
        # 使用torch.jit.script优化注意力计算
        @torch.jit.script
        def compute_attention(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            return torch.matmul(weights, values)
        
        # 计算注意力输出
        attn_output = compute_attention(attn_weights, v)
        
        # 重新排列维度并投影
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        
        if present is not None:
            return attn_output, present
        return attn_output


class KVCache:
    """高效的KV缓存管理
    
    基于deepseek-ai的FlashMLA项目，实现了高效的键值缓存管理，
    用于加速自回归生成过程。
    
    参考: https://github.com/deepseek-ai/FlashMLA
    """
    
    def __init__(self, max_batch_size, max_seq_len, n_layers, n_heads, head_dim):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        # 初始化缓存
        self.reset()
        
    def reset(self):
        """重置缓存"""
        self.key_cache = [None] * self.n_layers
        self.value_cache = [None] * self.n_layers
        self.current_seq_len = 0
        
    def update(self, layer_idx, k, v):
        """更新缓存
        
        Args:
            layer_idx: 层索引
            k: 键张量
            v: 值张量
        """
        batch_size = k.size(0)
        seq_len = k.size(1)
        
        # 如果缓存为空，初始化缓存
        if self.key_cache[layer_idx] is None:
            shape = (batch_size, self.max_seq_len, self.n_heads, self.head_dim)
            self.key_cache[layer_idx] = torch.zeros(shape, dtype=k.dtype, device=k.device)
            self.value_cache[layer_idx] = torch.zeros(shape, dtype=v.dtype, device=v.device)
        
        # 更新缓存
        end_pos = min(self.current_seq_len + seq_len, self.max_seq_len)
        if end_pos > self.current_seq_len:  # 只有在有新token时才更新
            self.key_cache[layer_idx][:, self.current_seq_len:end_pos] = k[:, :(end_pos-self.current_seq_len)]
            self.value_cache[layer_idx][:, self.current_seq_len:end_pos] = v[:, :(end_pos-self.current_seq_len)]
            
            # 更新当前序列长度
            if layer_idx == self.n_layers - 1:  # 最后一层更新完成后更新序列长度
                self.current_seq_len = end_pos
        
    def get(self, layer_idx):
        """获取缓存
        
        Args:
            layer_idx: 层索引
            
        Returns:
            当前层的键值缓存元组
        """
        if self.key_cache[layer_idx] is None:
            return None
        
        # 只返回有效部分的缓存
        k = self.key_cache[layer_idx][:, :self.current_seq_len]
        v = self.value_cache[layer_idx][:, :self.current_seq_len]
        
        return k, v


class FlashMHAModule(nn.Module):
    """FlashMLA高效多头注意力模块
    
    整合了多种优化技术的多头注意力模块，包括：
    1. Flash Attention - 高效的注意力计算
    2. 块稀疏注意力 - 降低计算复杂度
    3. 多查询注意力 - 减少内存使用
    4. KV缓存 - 加速自回归生成
    
    参考: https://github.com/deepseek-ai/FlashMLA
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
        self.use_kv_cache = attn_config.get('kv_cache', False)
        
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
        elif self.use_flash_attention:
            # 使用Flash Attention实现
            from .flash_attention import FlashAttention
            self.attn = FlashAttention(
                n_embd=self.n_embd,
                n_head=self.n_head,
                dropout_p=attn_config.get('attention_dropout', 0.1)
            )
        else:
            # 使用标准多头注意力
            from .attention import MultiHeadAttention
            self.attn = MultiHeadAttention(config)
        
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
    
    def forward(self, x, layer_idx=0, attention_mask=None):
        """前向传播
        
        Args:
            x: 输入张量
            layer_idx: 层索引（用于KV缓存）
            attention_mask: 注意力掩码
            
        Returns:
            注意力输出
        """
        # 获取KV缓存（如果启用）
        layer_past = None
        if self.use_kv_cache and self.kv_cache is not None:
            layer_past = self.kv_cache.get(layer_idx)
        
        # 应用注意力
        if layer_past is not None:
            output, present = self.attn(x, attention_mask=attention_mask, layer_past=layer_past)
            # 更新KV缓存
            if present is not None:
                self.kv_cache.update(layer_idx, *present)
            return output
        else:
            return self.attn(x, attention_mask=attention_mask)