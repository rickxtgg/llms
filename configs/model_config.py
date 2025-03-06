"""模型配置文件

这个文件包含了模型的核心参数配置，可以通过修改这些参数来调整模型的大小和性能。
基于deepseek-ai的开源项目（DualPipe、EPLB、FlashMLA等）实现了高效的训练和推理优化。
"""

class ModelConfig:
    """Transformer模型配置类"""
    
    def __init__(self):
        # 模型基本参数
        self.vocab_size = 50257  # GPT-2词表大小
        self.n_positions = 1024  # 最大序列长度
        self.n_embd = 768  # 嵌入维度
        self.n_layer = 12  # Transformer层数
        self.n_head = 12  # 注意力头数
        self.n_inner = self.n_embd * 4  # FFN内部维度
        
        # 激活函数和dropout
        self.activation_function = "gelu"
        self.resid_pdrop = 0.1  # 残差dropout
        self.embd_pdrop = 0.1  # 嵌入dropout
        self.attn_pdrop = 0.1  # 注意力dropout
        
        # 训练参数
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        
        # 训练设置
        self.max_steps = 100000
        self.warmup_steps = 10000
        self.batch_size = 32
        self.gradient_accumulation_steps = 1
        self.fp16 = True  # 混合精度训练
        
        # 数据参数
        self.train_file = "data/processed/train.txt"
        self.val_file = "data/processed/val.txt"
        self.test_file = "data/processed/test.txt"
        
        # 高级优化参数（基于deepseek-ai项目）
        # 内存优化（基于EPLB项目）
        self.memory_optimization = {
            "gradient_checkpointing": True,  # 梯度检查点，减少内存使用
            "activation_recomputation": False,  # 激活重计算
            "zero_optimization": False,  # ZeRO优化
            "zero_stage": 1,  # ZeRO优化阶段 (1, 2, 或 3)
            "offload_optimizer": False,  # 优化器状态卸载到CPU
            "offload_param": False  # 参数卸载到CPU
        }
        
        # 分布式训练（基于DualPipe项目）
        self.distributed_training = {
            "data_parallel": True,  # 数据并行
            "model_parallel": False,  # 模型并行
            "pipeline_parallel": False,  # 流水线并行
            "pipeline_chunks": 1,  # 流水线微批次数量
            "tensor_parallel": False,  # 张量并行
            "sequence_parallel": False  # 序列并行
        }
        
        # 注意力优化（基于FlashMLA项目）
        self.attention_optimization = {
            "flash_attention": True,  # 使用Flash Attention
            "rotary_position_embeddings": False,  # 使用旋转位置编码
            "attention_scale": 1.0 / (self.n_embd ** 0.5),  # 注意力缩放因子
            "kv_cache": True  # 使用KV缓存加速推理
        }
        
    def __repr__(self):
        return f"ModelConfig(n_layer={self.n_layer}, n_head={self.n_head}, n_embd={self.n_embd})"


# 不同大小的模型配置
def get_small_config():
    """获取小型模型配置 (~125M参数)"""
    config = ModelConfig()
    config.n_layer = 12
    config.n_head = 12
    config.n_embd = 768
    return config

def get_medium_config():
    """获取中型模型配置 (~350M参数)"""
    config = ModelConfig()
    config.n_layer = 24
    config.n_head = 16
    config.n_embd = 1024
    return config

def get_large_config():
    """获取大型模型配置 (~760M参数)"""
    config = ModelConfig()
    config.n_layer = 36
    config.n_head = 20
    config.n_embd = 1280
    return config

# 默认配置
default_config = get_small_config()