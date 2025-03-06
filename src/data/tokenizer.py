"""分词器模块

这个模块提供了文本分词功能，用于将原始文本转换为模型可以处理的token序列。
"""

import os
import json
import logging
from typing import List, Dict, Union, Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

class SimpleTokenizer:
    """简单的分词器实现，用于演示和测试"""
    
    def __init__(self, vocab_file: Optional[str] = None):
        """初始化分词器
        
        Args:
            vocab_file: 词表文件路径，如果为None则使用默认词表
        """
        self.token_to_id = {}
        self.id_to_token = {}
        
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        else:
            # 创建一个简单的默认词表（仅用于演示）
            self._create_default_vocab()
            
        logger.info(f"分词器初始化完成，词表大小: {len(self.token_to_id)}")
    
    def _load_vocab(self, vocab_file: str):
        """从文件加载词表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            
        for token, token_id in vocab.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
    
    def _create_default_vocab(self):
        """创建默认词表（ASCII字符）"""
        # 添加特殊token
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
        # 添加ASCII字符
        for i in range(128):
            char = chr(i)
            if char not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[char] = token_id
                self.id_to_token[token_id] = char
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token ID列表
        
        Args:
            text: 输入文本
            
        Returns:
            token ID列表
        """
        # 简单实现：按字符分割
        tokens = list(text)
        ids = []
        
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # 对于未知token，使用<unk>的ID
                ids.append(self.token_to_id["<unk>"])
                
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """将token ID列表解码为文本
        
        Args:
            ids: token ID列表
            
        Returns:
            解码后的文本
        """
        tokens = []
        
        for token_id in ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(self.id_to_token[self.token_to_id["<unk>"]])
                
        return "".join(tokens)
    
    def save(self, output_file: str):
        """保存词表到文件
        
        Args:
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
            
        logger.info(f"词表已保存到 {output_file}")



def get_tokenizer(tokenizer_path: Optional[str] = None) -> Union[PreTrainedTokenizer, SimpleTokenizer]:
    """获取分词器
    
    Args:
        tokenizer_path: 分词器路径，如果为None则使用默认分词器
        
    Returns:
        加载的分词器
    """
    if tokenizer_path and os.path.exists(tokenizer_path):
        logger.info(f"从 {tokenizer_path} 加载分词器")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.info("使用简单的默认分词器")
        tokenizer = SimpleTokenizer()
        
    return tokenizer



if __name__ == "__main__":
    # 简单测试
    tokenizer = SimpleTokenizer()
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    
    print(f"原始文本: {text}")
    print(f"编码后: {ids}")
    print(f"解码后: {decoded_text}")