"""训练脚本

这个脚本用于训练大语言模型，包括加载配置、初始化模型、准备数据集和启动训练过程。
"""

import os
import sys
import logging
import argparse
from importlib import import_module

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import GPT
from src.training.trainer import Trainer
from src.data.preprocessing import TextDataset
from src.data.tokenizer import get_tokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练大语言模型")
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
        help="配置变量名，例如 'default_config', 'get_small_config()'"
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="训练数据文件路径，如果不提供则使用配置中的路径"
    )
    parser.add_argument(
        "--val_file", 
        type=str, 
        default=None, 
        help="验证数据文件路径，如果不提供则使用配置中的路径"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=None, 
        help="分词器路径，如果不提供则使用默认分词器"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="checkpoints", 
        help="模型输出目录"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None, 
        help="从检查点恢复训练"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="随机种子"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        choices=["fp16", "fp32", "fp64", "bf16"], 
        default=None, 
        help="训练精度类型: fp16(混合精度), fp32(单精度), fp64(双精度), bf16(bfloat16)"
    )
    parser.add_argument(
        "--memory_efficient", 
        action="store_true", 
        help="启用内存高效训练（梯度检查点、激活重计算等）"
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
        
        # 处理函数调用，如 'get_small_config()' 或 'get_small_config'
        if '()' in config_name:
            # 如果名称中包含括号，移除括号并调用函数
            config_func = getattr(config_module, config_name.replace('()', ''))
            config = config_func()
        else:
            # 检查获取的属性是否为可调用函数
            attr = getattr(config_module, config_name)
            if callable(attr):
                # 如果是函数，调用它
                config = attr()
            else:
                # 如果不是函数，直接使用该属性
                config = attr
            
        return config
    except (ImportError, AttributeError) as e:
        logger.error(f"加载配置失败: {e}")
        sys.exit(1)


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
    if args.train_file:
        config.train_file = args.train_file
    if args.val_file:
        config.val_file = args.val_file
        
    # 更新精度设置（如果在命令行中提供）
    if args.precision:
        config.precision = args.precision
        # 根据精度类型设置相应的标志
        config.fp16 = (args.precision == "fp16")
        config.fp64 = (args.precision == "fp64")
        
    # 更新内存优化设置（如果启用）
    if args.memory_efficient:
        if not hasattr(config, 'memory_optimization'):
            config.memory_optimization = {}
        config.memory_optimization["gradient_checkpointing"] = True
        config.memory_optimization["activation_recomputation"] = True
        config.memory_optimization["memory_efficient_attention"] = True
        config.memory_optimization["recompute_activation"] = True
        config.memory_optimization["pin_memory"] = True
        config.memory_optimization["optimize_device_placement"] = True
    
    # 获取分词器
    tokenizer = get_tokenizer(args.tokenizer_path)
    
    # 准备数据集
    logger.info(f"加载训练数据集: {config.train_file}")
    train_dataset = TextDataset(
        file_path=config.train_file,
        tokenizer=tokenizer,
        block_size=config.n_positions,
        shuffle=True
    )
    
    val_dataset = None
    if hasattr(config, 'val_file') and os.path.exists(config.val_file):
        logger.info(f"加载验证数据集: {config.val_file}")
        val_dataset = TextDataset(
            file_path=config.val_file,
            tokenizer=tokenizer,
            block_size=config.n_positions,
            shuffle=False
        )
    
    # 初始化模型
    logger.info("初始化模型")
    model = GPT(config)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数量: {total_params:,}")
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # 从检查点恢复（如果提供）
    if args.resume_from_checkpoint:
        logger.info(f"从检查点 {args.resume_from_checkpoint} 恢复训练")
        trainer.load_model(args.resume_from_checkpoint)
    
    # 开始训练
    logger.info("开始训练")
    global_step, best_val_loss = trainer.train()
    
    logger.info(f"训练完成，全局步数: {global_step}, 最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()