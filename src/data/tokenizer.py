"""分词器模块

这个模块提供了文本分词功能，用于将原始文本转换为模型可以处理的token序列。
"""

import os
import json
import logging
from typing import List, Dict, Union, Optional

import torch
import jieba
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

    def encode(self, text: str) -> List[int]:
        """将输入文本转换为token ID序列
        
        Args:
            text: 输入文本
        
        Returns:
            token ID序列
        """
        tokens = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.token_to_id["<unk>"])
        return tokens
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
        """创建默认词表（包含ASCII字符和常用中文字符）"""
        # 添加特殊token
        special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
        # 添加ASCII字符
        current_id = len(special_tokens)
        for i in range(32, 127):
            char = chr(i)
            self.token_to_id[char] = current_id
            self.id_to_token[current_id] = char
            current_id += 1
            
        # 添加常用中文字符（基本汉字范围：0x4E00-0x9FFF）
        for i in range(0x4E00, 0x9FFF):
            char = chr(i)
            self.token_to_id[char] = current_id
            self.id_to_token[current_id] = char
            current_id += 1

def get_tokenizer(tokenizer_path: Optional[str] = None) -> Union[PreTrainedTokenizer, SimpleTokenizer]:
    """获取分词器
    
    Args:
        tokenizer_path: 分词器路径，如果为None则使用默认分词器
        
    Returns:
        加载的分辞器
    """
    if tokenizer_path and os.path.exists(tokenizer_path):
        logger.info(f"从 {tokenizer_path} 加载分词器")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.info("使用简单的默认分词器")
        tokenizer = SimpleTokenizer()
        
    return tokenizer

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

    def encode(self, text: str) -> List[int]:
        """将输入文本转换为token ID序列
        
        Args:
            text: 输入文本
        
        Returns:
            token ID序列
        """
        tokens = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.token_to_id["<unk>"])
        return tokens
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
        """创建默认词表（包含ASCII字符和常用中文字符）"""
        # 添加特殊token
        special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
        # 添加ASCII字符
        current_id = len(special_tokens)
        for i in range(32, 127):
            char = chr(i)
            self.token_to_id[char] = current_id
            self.id_to_token[current_id] = char
            current_id += 1
            
        # 添加常用中文字符（基本汉字范围：0x4E00-0x9FFF）
        for i in range(0x4E00, 0x9FFF):
            char = chr(i)
            self.token_to_id[char] = current_id
            self.id_to_token[current_id] = char
            current_id += 1