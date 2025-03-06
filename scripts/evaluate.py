"""评估脚本

这个脚本用于评估训练好的大语言模型，包括加载模型、准备测试数据集和计算评估指标。
"""

import os
import sys
import logging
import argparse
import math
from importlib import import_module

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import GPT
from src.data.preprocessing import TextDataset
from src.data.tokenizer import get_tokenizer
from src.evaluation.metrics import calculate_perplexity, calculate_accuracy


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估大语言模型")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="模型权重文件路径"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs.model_config", 
        help="配置模块路径，例如 'configs.model_config'"
    )
    parser.add_argument(
        "--config_name", 
        type=str, 
        default="default_config", 
        help="配置变量名，例如 'default_config'"
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default=None, 
        help="测试数据文件路径，如果不提供则使用配置中的路径"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=None, 
        help="分词器路径，如果不提供则使用默认分词器"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="评估批次大小"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None, 
        help="最大评估样本数，如果为None则使用全部测试集"
    )
    parser.add_argument(
        "--generate_samples", 
        type=int, 
        default=5, 
        help="生成样本数量"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="随机种子"
    )
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path, config_name):
    """加载模型配置"""
    try:
        config_module = import_module(config_path)
        
        # 处理函数调用，如 'get_small_config()'
        if '()' in config_name:
            config_func = getattr(config_module, config_name.replace('()', ''))
            config = config_func()
        else:
            config = getattr(config_module, config_name)
            
        return config
    except (ImportError, AttributeError) as e:
        logger.error(f"加载配置失败: {e}")
        sys.exit(1)


def load_model(model_path, config):
    """加载模型"""
    model = GPT(config)
    
    # 加载模型权重
    if os.path.isdir(model_path):
        # 如果是目录，尝试加载model.pt文件
        model_path = os.path.join(model_path, "model.pt")
        
    if not os.path.exists(model_path):
        logger.error(f"模型文件 {model_path} 不存在")
        sys.exit(1)
        
    logger.info(f"从 {model_path} 加载模型")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    return model


def evaluate_model(model, test_dataset, batch_size=16, max_samples=None):
    """评估模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    total_loss = 0
    total_samples = 0
    sample_count = 0
    
    logger.info("开始评估...")
    with torch.no_grad():
        for batch in test_dataloader:
            # 检查是否达到最大样本数
            if max_samples is not None and sample_count >= max_samples:
                break
                
            # 将数据移动到设备
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播和损失计算
            outputs, loss = model(input_ids, targets=labels)
            
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            sample_count += batch_size
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)
    
    logger.info(f"评估完成，样本数: {total_samples}")
    logger.info(f"平均损失: {avg_loss:.4f}")
    logger.info(f"困惑度: {perplexity:.4f}")
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "samples": total_samples
    }


def generate_text_samples(model, tokenizer, num_samples=5, max_length=100, temperature=1.0, top_k=50):
    """生成文本样本"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    logger.info(f"生成 {num_samples} 个文本样本...")
    
    # 生成样本
    for i in range(num_samples):
        # 使用简单的提示
        prompt = "人工智能是"
        input_ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # 生成文本
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k
            )
            
        # 解码生成的文本
        generated_text = tokenizer.decode(output_ids[0].tolist())
        
        logger.info(f"样本 {i+1}:\n{generated_text}\n")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    logger.info(f"从 {args.config} 加载配置 {args.config_name}")
    config = load_config(args.config, args.config_name)
    
    # 更新配置中的文件路径（如果在命令行中提供）
    if args.test_file:
        config.test_file = args.test_file
    
    # 获取分词器
    tokenizer = get_tokenizer(args.tokenizer_path)
    
    # 加载模型
    model = load_model(args.model_path, config)
    
    # 准备测试数据集
    logger.info(f"加载测试数据集: {config.test_file}")
    test_dataset = TextDataset(
        file_path=config.test_file,
        tokenizer=tokenizer,
        block_size=config.n_positions,
        shuffle=False
    )
    
    # 评估模型
    results = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # 生成文本样本
    if args.generate_samples > 0:
        generate_text_samples(
            model=model,
            tokenizer=tokenizer,
            num_samples=args.generate_samples
        )


if __name__ == "__main__":
    main()