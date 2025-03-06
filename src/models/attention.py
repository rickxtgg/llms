"""注意力机制模块

这个模块实现了Transformer架构中的多头自注意力机制，是大语言模型的核心组件之一。
基于deepseek-ai的FlashMLA项目，实现了高效的注意力计算机制，显著提高了训练和推理效率。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """单头注意力机制
    
    基于deepseek-ai的FlashMLA项目，实现了高效的注意力计算。
    """
    
    def __init__(self, n_embd, head_size, n_positions, attn_pdrop=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(attn_pdrop)
        
        # 注册缓冲区用于存储注意力掩码
        mask = torch.tril(torch.ones(n_positions, n_positions))
        self.register_buffer("mask", mask.view(1, 1, n_positions, n_positions))
        
        # 是否使用KV缓存加速推理
        self.use_kv_cache = False
        
    def forward(self, x, layer_past=None, attention_mask=None):
        B, T, C = x.size()  # batch_size, sequence_length, embedding_dim
        
        # 计算查询、键、值
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        # 处理KV缓存（用于加速推理）
        if layer_past is not None and self.use_kv_cache:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)
            present = (k, v)
        else:
            present = None
            
        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, T, T或K)
        
        # 应用掩码（解码器中的自回归属性）
        if layer_past is not None and self.use_kv_cache:
            # 对于推理，只需要掩码最后一个位置
            att = att[:, :, -1:, :]
        else:
            # 对于训练，应用完整掩码
            att = att.masked_fill(self.mask[:, :, :T, :k.size(1)] == 0, float('-inf'))
        
        # 应用外部注意力掩码（如果提供）
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        
        # 对注意力权重进行softmax归一化
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 加权聚合值向量
        y = att @ v  # (B, T, head_size) 或 (B, 1, head_size)
        y = self.resid_dropout(y)
        
        if present is not None:
            return y, present
        return y


class MultiHeadAttention(nn.Module):
    """多头注意力机制
    
    基于deepseek-ai的FlashMLA项目，实现了高效的多头注意力计算。
    支持Flash Attention和KV缓存等优化技术。
    """
    
    def __init__(self, config):
        super().__init__()
        n_embd = config.n_embd
        self.n_head = config.n_head
        assert n_embd % self.n_head == 0, "嵌入维度必须能被注意力头数整除"
        
        self.head_size = n_embd // self.n_head
        self.head_dim = self.head_size  # 添加head_dim作为head_size的别名，保持一致性
        
        # 获取注意力优化配置
        self.use_flash_attention = getattr(config, 'attention_optimization', {}).get('flash_attention', False)
        self.use_rotary_pos_emb = getattr(config, 'attention_optimization', {}).get('rotary_position_embeddings', False)
        self.use_kv_cache = getattr(config, 'attention_optimization', {}).get('kv_cache', False)
        
        if self.use_flash_attention:
            # 使用优化的Flash Attention实现
            self.query = nn.Linear(n_embd, n_embd)
            self.key = nn.Linear(n_embd, n_embd)
            self.value = nn.Linear(n_embd, n_embd)
        else:
            # 使用传统的多头实现
            self.heads = nn.ModuleList([
                AttentionHead(n_embd, self.head_size, config.n_positions, config.attn_pdrop) 
                for _ in range(self.n_head)
            ])
            # 设置KV缓存
            if self.use_kv_cache:
                for head in self.heads:
                    head.use_kv_cache = True
        
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self, x, layer_past=None, attention_mask=None):
        if self.use_flash_attention:
            # 使用优化的Flash Attention实现
            B, T, C = x.size()
            
            # 计算查询、键、值
            q = self.query(x).view(B, T, self.n_head, self.head_dim)
            k = self.key(x).view(B, T, self.n_head, self.head_dim)
            v = self.value(x).view(B, T, self.n_head, self.head_dim)
            
            # 处理KV缓存
            present = None
            if layer_past is not None and self.use_kv_cache:
                past_k, past_v = layer_past
                k = torch.cat((past_k, k), dim=1)
                v = torch.cat((past_v, v), dim=1)
                present = (k, v)
            
            # 应用旋转位置编码（如果启用）
            if self.use_rotary_pos_emb:
                from .utils import get_rotary_pos_emb, apply_rotary_pos_emb
                cos, sin = get_rotary_pos_emb(T, self.head_dim, x.device)
                q = apply_rotary_pos_emb(q, cos, sin)
                k = apply_rotary_pos_emb(k, cos, sin)
            
            # 创建注意力掩码
            mask = None
            if layer_past is None or not self.use_kv_cache:  # 训练模式
                # 创建因果掩码
                mask = torch.tril(torch.ones(T, k.size(1), device=x.device)).view(1, 1, T, k.size(1))
            
            # 如果提供了外部掩码，与因果掩码结合
            if attention_mask is not None:
                if mask is not None:
                    mask = mask & attention_mask
                else:
                    mask = attention_mask
            
            # 使用Flash Attention计算
            from .utils import flash_attention
            attn_output = flash_attention(q, k, v, mask)
            
            # 重塑输出并投影
            attn_output = attn_output.reshape(B, T, C)
            attn_output = self.proj(attn_output)
            attn_output = self.resid_dropout(attn_output)
            
            if present is not None:
                return attn_output, present
            return attn_output
        else:
            # 使用传统的多头实现
            if layer_past is not None and self.use_kv_cache:
                # 处理推理阶段的KV缓存
                outputs = []
                presents = []
                for i, head in enumerate(self.heads):
                    output, present = head(x, layer_past[i] if isinstance(layer_past, list) else None, attention_mask)
                    outputs.append(output)
                    presents.append(present)
                out = torch.cat(outputs, dim=-1)
                out = self.proj(out)
                out = self.resid_dropout(out)
                return out, presents
            else:
                # 训练阶段
                out = torch.cat([h(x, attention_mask=attention_mask) for h in self.heads], dim=-1)
                out = self.proj(out)
                out = self.resid_dropout(out)
                return out


class SelfAttention(nn.Module):
    """带有残差连接和层归一化的自注意力层
    
    基于deepseek-ai的FlashMLA项目，支持高效的注意力计算和KV缓存。
    """
    
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ln = nn.LayerNorm(config.n_embd)
        
        # 是否使用梯度检查点（用于内存优化）
        self.gradient_checkpointing = getattr(config, 'memory_optimization', {}).get('gradient_checkpointing', False)
        
    def forward(self, x, layer_past=None, attention_mask=None):
        # 获取归一化输入
        norm_x = self.ln(x)
        
        # 应用注意力，处理KV缓存
        if layer_past is not None:
            attn_output, present = self.attn(norm_x, layer_past, attention_mask)
            # 应用残差连接
            output = x + attn_output
            return output, present
        else:
            # 使用梯度检查点以节省内存（如果启用）
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                # 修复：确保在梯度检查点中正确处理attention_mask
                if attention_mask is not None:
                    attn_output = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.attn),
                        norm_x,
                        attention_mask=attention_mask
                    )
                else:
                    attn_output = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.attn),
                        norm_x
                    )
            else:
                # 直接调用注意力层
                attn_output = self.attn(norm_x, attention_mask=attention_mask)
            
            # 应用残差连接
            return x + attn_output