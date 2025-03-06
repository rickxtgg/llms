"""Transformer模型实现

这个模块实现了基于Transformer架构的语言模型，包括嵌入层、位置编码、多层自注意力和前馈网络等组件。
基于deepseek-ai的开源项目（DualPipe、EPLB、FlashMLA等）实现了高效的训练和推理优化。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention


class MLP(nn.Module):
    """多层感知机，用于Transformer的前馈网络"""
    
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_inner)
        self.proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer解码器层
    
    基于deepseek-ai的FlashMLA项目，支持KV缓存和梯度检查点等优化技术。
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
        # 是否使用梯度检查点（用于内存优化）
        self.gradient_checkpointing = getattr(config, 'memory_optimization', {}).get('gradient_checkpointing', False)
        
    def forward(self, x, layer_past=None, use_cache=False):
        # 处理KV缓存
        if layer_past is not None or use_cache:
            # 自注意力层（带KV缓存）
            attn_output = self.ln1(x)
            attn_output, present = self.attn(attn_output, layer_past)
            x = x + attn_output
            # 前馈网络层
            x = x + self.mlp(self.ln2(x))
            if use_cache:
                return x, present
            return x
        else:
            # 使用梯度检查点以节省内存（如果启用）
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                # 自注意力层
                attn_output = self.ln1(x)
                attn_output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.attn), attn_output)
                x = x + attn_output
                
                # 前馈网络层
                mlp_output = self.ln2(x)
                mlp_output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mlp), mlp_output)
                x = x + mlp_output
            else:
                # 自注意力层
                x = x + self.attn(self.ln1(x))
                # 前馈网络层
                x = x + self.mlp(self.ln2(x))
            return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 添加位置编码到输入
        return x + self.pe[:, :x.size(1), :]


class GPT(nn.Module):
    """基于Transformer的GPT模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置编码
        self.pos_enc = PositionalEncoding(config.n_embd, config.n_positions)
        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        # Transformer层
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 输出投影
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, targets=None):
        batch_size, seq_length = input_ids.size()
        
        # 获取词嵌入
        token_embeddings = self.tok_emb(input_ids)  # [batch_size, seq_length, n_embd]
        
        # 添加位置编码
        x = self.pos_enc(token_embeddings)
        x = self.drop(x)
        
        # 应用Transformer层
        for block in self.blocks:
            x = block(x)
            
        # 应用最终层归一化
        x = self.ln_f(x)
        
        # 计算logits
        logits = self.head(x)  # [batch_size, seq_length, vocab_size]
        
        # 如果提供了目标，计算损失
        loss = None
        if targets is not None:
            # 重塑logits以适应CrossEntropyLoss
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """生成文本"""
        # 初始化KV缓存
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # 如果序列太长，截断
            if input_ids.size(1) > self.config.n_positions:
                input_ids = input_ids[:, -self.config.n_positions:]
                past_key_values = None  # 重置KV缓存
                
            # 前向传播
            with torch.no_grad():
                # 使用KV缓存加速生成
                if past_key_values is None:
                    # 首次计算，处理整个序列
                    logits, _ = self.forward(input_ids)
                    # 获取最后一个时间步的logits
                    logits = logits[:, -1, :] / temperature
                else:
                    # 后续计算，只处理最新的token
                    current_token = input_ids[:, -1].unsqueeze(1)  # 只取最后一个token
                    
                    # 获取词嵌入
                    token_embeddings = self.tok_emb(current_token)  # [batch_size, 1, n_embd]
                    
                    # 添加位置编码（注意位置索引）
                    position_index = input_ids.size(1) - 1
                    position_embedding = self.pos_enc.pe[:, position_index:position_index+1, :]
                    x = token_embeddings + position_embedding
                    x = self.drop(x)
                    
                    # 应用Transformer层，使用KV缓存
                    presents = []
                    for i, block in enumerate(self.blocks):
                        x, present = block(x, past_key_values[i], use_cache=True)
                        presents.append(present)
                    
                    # 更新KV缓存
                    past_key_values = presents
                    
                    # 应用最终层归一化
                    x = self.ln_f(x)
                    
                    # 计算logits
                    logits = self.head(x)  # [batch_size, 1, vocab_size]
                    logits = logits.squeeze(1) / temperature
            
            # 可选：应用top-k过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 将新token添加到输入序列
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
        return input_ids