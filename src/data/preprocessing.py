"""数据预处理模块

这个模块提供了处理原始文本数据的功能，包括数据清洗、分割和格式化等操作。
"""

import os
import json
import logging
import random
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """文本数据集类，用于加载和处理文本数据"""
    
    def __init__(self, file_path: str, tokenizer, block_size: int = 1024, shuffle: bool = True):
        """
        初始化文本数据集
        
        Args:
            file_path: 文本文件路径
            tokenizer: 分词器
            block_size: 文本块大小
            shuffle: 是否打乱数据
        """
        assert os.path.isfile(file_path), f"输入文件 {file_path} 不存在"
        
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        
        logger.info(f"从 {file_path} 加载数据")
        self._load_and_process_data(file_path)
        
        if shuffle:
            random.shuffle(self.examples)
            
        logger.info(f"数据集加载完成，共有 {len(self.examples)} 个样本")
    
    def _load_and_process_data(self, file_path: str):
        """加载并处理数据"""
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 使用分词器处理文本
        tokenized_text = self.tokenizer.encode(text)
        
        # 将文本分割成固定大小的块
        for i in range(0, len(tokenized_text) - self.block_size, self.block_size):
            example = {
                "input_ids": tokenized_text[i:i + self.block_size],
                "labels": tokenized_text[i:i + self.block_size],
            }
            self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.examples[idx]["input_ids"], dtype=torch.long),
            "labels": torch.tensor(self.examples[idx]["labels"], dtype=torch.long),
        }


def prepare_dataset(raw_data_dir: str, processed_data_dir: str, tokenizer_path: Optional[str] = None):
    """准备数据集
    
    Args:
        raw_data_dir: 原始数据目录
        processed_data_dir: 处理后数据保存目录
        tokenizer_path: 分词器路径，如果为None则使用默认分词器
    """
    # 确保目录存在
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # 获取分词器
    tokenizer = get_tokenizer(tokenizer_path)
    
    # 处理训练、验证和测试数据
    for split in ["train", "val", "test"]:
        input_file = os.path.join(raw_data_dir, f"{split}.txt")
        if not os.path.exists(input_file):
            logger.warning(f"文件 {input_file} 不存在，跳过处理")
            continue
            
        output_file = os.path.join(processed_data_dir, f"{split}.txt")
        
        logger.info(f"处理 {split} 数据...")
        
        # 读取原始文本
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 清洗文本（可以根据需要添加更多清洗步骤）
        text = clean_text(text)
        
        # 保存处理后的文本
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
            
        logger.info(f"{split} 数据处理完成，保存到 {output_file}")


def clean_text(text: str) -> str:
    """清洗文本
    
    Args:
        text: 输入文本
        
    Returns:
        清洗后的文本
    """
    # 移除多余的空白字符
    text = ' '.join(text.split())
    
    # 这里可以添加更多的文本清洗步骤，如：
    # - 移除HTML标签
    # - 规范化标点符号
    # - 处理特殊字符
    # - 等等
    
    return text


def main():
    """主函数，用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据预处理工具")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="原始数据目录")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="处理后数据保存目录")
    parser.add_argument("--tokenizer", type=str, default=None, help="分词器路径")
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    
    # 准备数据集
    prepare_dataset(args.raw_dir, args.processed_dir, args.tokenizer)


if __name__ == "__main__":
    main()