"""训练器模块

这个模块实现了大语言模型的训练流程，包括数据加载、优化器配置、训练循环和模型保存等功能。
"""

import os
import time
import math
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer:
    """大语言模型训练器"""
    
    def __init__(self, model, config, train_dataset, val_dataset=None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 模型配置
            train_dataset: 训练数据集
            val_dataset: 验证数据集
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 设置精度训练
        self.precision = getattr(config, 'precision', 'fp32')
        self.fp16 = getattr(config, 'fp16', False)
        self.fp64 = getattr(config, 'fp64', False)
        
        # 设置混合精度训练的梯度缩放器
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        
        # 如果使用双精度训练，将模型转换为float64
        if self.fp64:
            self.model = self.model.double()
            
        # 应用内存优化
        self._apply_memory_optimizations()
        
    def _apply_memory_optimizations(self):
        """应用内存优化技术"""
        # 获取内存优化配置
        memory_config = getattr(self.config, 'memory_optimization', {})
        
        # 应用梯度检查点
        if memory_config.get('gradient_checkpointing', False):
            # 为模型中的Transformer块启用梯度检查点
            for name, module in self.model.named_modules():
                if "blocks" in name and isinstance(module, nn.Module):
                    if hasattr(module, 'gradient_checkpointing'):
                        module.gradient_checkpointing = True
            
            # 如果模型有全局梯度检查点属性，也启用它
            if hasattr(self.model, 'gradient_checkpointing'):
                self.model.gradient_checkpointing = True
        
        # 应用激活重计算
        if memory_config.get('activation_recomputation', False):
            for name, module in self.model.named_modules():
                if "attention" in name.lower() and hasattr(module, 'save_attention_output'):
                    module.save_attention_output = False
        
        # 使用固定内存加速CPU-GPU传输
        if memory_config.get('pin_memory', True):
            torch.cuda.empty_cache()
            
        # 优化设备放置
        if memory_config.get('optimize_device_placement', True):
            if torch.cuda.is_available():
                torch.cuda.set_device(torch.cuda.current_device())
                
        # 如果启用了CPU卸载，设置相关参数
        if memory_config.get('cpu_offload', False):
            # 这里可以添加将不活跃参数卸载到CPU的逻辑
            pass
        
        # 设置TensorBoard
        self.writer = SummaryWriter(log_dir="runs")
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def _create_optimizer(self):
        """创建优化器"""
        # 将权重衰减应用于所有参数，除了偏置和LayerNorm参数
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
    
    def _create_scheduler(self):
        """创建学习率调度器，包括预热和线性衰减"""
        def lr_lambda(current_step):
            # 预热阶段
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            # 线性衰减
            return max(
                0.0,
                float(self.config.max_steps - current_step) / 
                float(max(1, self.config.max_steps - self.config.warmup_steps))
            )
            
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """执行完整的训练流程"""
        logger.info("***** 开始训练 *****")
        logger.info(f"  模型: {self.config}")
        logger.info(f"  训练数据集大小: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"  验证数据集大小: {len(self.val_dataset)}")
        logger.info(f"  批次大小: {self.config.batch_size}")
        logger.info(f"  梯度累积步数: {self.config.gradient_accumulation_steps}")
        logger.info(f"  总训练步数: {self.config.max_steps}")
        
        # 检查数据集大小并调整批次大小
        dataset_size = len(self.train_dataset)
        if dataset_size < self.config.batch_size:
            logger.warning(f"数据集大小({dataset_size})小于批次大小({self.config.batch_size})，自动调整批次大小")
            self.config.batch_size = max(1, dataset_size // 2)
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,  # 不丢弃最后一个不完整的批次
            num_workers=4,
            persistent_workers=True  # 保持worker进程存活
        )
        
        # 检查数据加载器是否正确初始化
        try:
            next(iter(train_dataloader))
        except Exception as e:
            raise RuntimeError(f"数据加载器初始化失败：{str(e)}")
            
        # 训练循环
        self.model.train()
        accumulated_loss = 0
        
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            if self.epoch % 1 == 0:
                logger.info(f"开始 Epoch {self.epoch}")
                epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {self.epoch}")
            else:
                epoch_iterator = train_dataloader
            
            for step, batch in enumerate(epoch_iterator):
                # 将数据移动到设备
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播和损失计算
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs, loss = self.model(input_ids, targets=labels)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # 反向传播
                if self.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 更新累积损失
                accumulated_loss += loss.item()
                
                # 梯度累积
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    # 更新学习率
                    self.scheduler.step()
                    
                    # 清零梯度
                    self.optimizer.zero_grad()
                    
                    # 更新全局步数
                    self.global_step += 1
                    
                    # 记录训练信息
                    if self.global_step % 10 == 0:
                        self.writer.add_scalar('train/loss', accumulated_loss, self.global_step)
                        self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                        accumulated_loss = 0
                    
                    # 验证
                    if self.val_dataset and self.global_step % 1000 == 0:
                        val_loss = self.evaluate()
                        self.writer.add_scalar('val/loss', val_loss, self.global_step)
                        
                        # 保存最佳模型
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_model("best_model")
                        
                        # 恢复训练模式
                        self.model.train()
                    
                    # 保存检查点
                    if self.global_step % 5000 == 0:
                        self.save_model(f"checkpoint-{self.global_step}")
                    
                    # 检查是否达到最大步数
                    if self.global_step >= self.config.max_steps:
                        epoch_iterator.close()
                        break
            
            # 检查是否达到最大步数
            if self.global_step >= self.config.max_steps:
                break
        
        # 保存最终模型
        self.save_model("final_model")
        logger.info("***** 训练完成 *****")
        
        # 关闭TensorBoard写入器
        self.writer.close()
        
        return self.global_step, self.best_val_loss
    
    def evaluate(self):
        """在验证集上评估模型"""
        if not self.val_dataset:
            return 0.0
        
        logger.info("***** 运行评估 *****")
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True
        )
        
        # 切换到评估模式
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="评估"):
                # 将数据移动到设备
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播和损失计算
                outputs, loss = self.model(input_ids, targets=labels)
                
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss)
        
        logger.info(f"验证损失: {avg_loss:.4f}, 困惑度: {perplexity:.4f}")
        
        return avg_loss
    
    def save_model(self, name):
        """保存模型和训练状态"""
        # 创建保存目录
        os.makedirs("checkpoints", exist_ok=True)
        save_path = os.path.join("checkpoints", name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), os.path.join(save_path, "model.pt"))
        
        # 保存训练状态
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, os.path.join(save_path, "trainer_state.pt"))
        
        logger.info(f"模型保存到 {save_path}")
        
    def load_model(self, path):
        """加载模型和训练状态"""
        # 加载模型权重
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        
        # 加载训练状态
        state = torch.load(os.path.join(path, "trainer_state.pt"))
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        if self.scaler and state['scaler']:
            self.scaler.load_state_dict(state['scaler'])
        self.global_step = state['global_step']
        self.epoch = state['epoch']
        self.best_val_loss = state['best_val_loss']
        
        logger.info(f"模型从 {path} 加载完成，全局步数: {self.global_step}")