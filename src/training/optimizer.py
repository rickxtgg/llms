"""优化器模块

这个模块提供了大语言模型训练中使用的优化器和学习率调度器的实现。
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class AdamWWithDecay(Optimizer):
    """实现带权重衰减的AdamW优化器
    
    这个优化器基于论文 "Decoupled Weight Decay Regularization" 实现，
    它将权重衰减与自适应学习率分离，提高了模型的泛化能力。
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, correct_bias=True):
        """初始化AdamW优化器
        
        Args:
            params: 要优化的参数
            lr: 学习率
            betas: Adam优化器的beta参数 (默认: (0.9, 0.999))
            eps: 数值稳定性常数 (默认: 1e-8)
            weight_decay: 权重衰减系数 (默认: 0.0)
            correct_bias: 是否校正偏差 (默认: True)
        """
        if lr < 0.0:
            raise ValueError(f"学习率不能为负数: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Beta参数错误: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Beta参数错误: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon不能为负数: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"权重衰减不能为负数: {weight_decay}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """执行单步优化
        
        Args:
            closure: 重新评估模型并返回损失的闭包 (默认: None)
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # 获取梯度
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW不支持稀疏梯度")
                    
                # 获取参数状态
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                # 获取超参数
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                # 更新步数
                state['step'] += 1
                
                # 应用权重衰减
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    
                # 更新一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group['lr']
                    
                # 应用更新
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """创建带预热的线性学习率调度器
    
    Args:
        optimizer: 要调度的优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        last_epoch: 上一轮的epoch (默认: -1)
        
    Returns:
        学习率调度器
    """
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 线性衰减
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
        
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """创建带预热的余弦学习率调度器
    
    Args:
        optimizer: 要调度的优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        num_cycles: 余弦周期数 (默认: 0.5)
        last_epoch: 上一轮的epoch (默认: -1)
        
    Returns:
        学习率调度器
    """
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦衰减
        progress = float(current_step - num_warmup_steps) / \
                  float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)