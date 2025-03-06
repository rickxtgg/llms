# 从零开始训练大语言模型

这个项目提供了从零开始训练大语言模型(LLM)的完整流程，包括数据处理、模型构建、训练和评估等关键步骤。本项目实现了基于Transformer架构的语言模型，并集成了deepseek-ai的多个开源项目的优化技术，显著提高了训练和推理效率。

## 适合训练的模型类型

本项目主要适合训练以下类型的模型：

1. **自回归语言模型**：类似GPT系列的自回归生成式大语言模型，适用于文本生成、对话系统、内容创作等任务。

2. **编码器-解码器模型**：通过修改配置，可以训练类似T5的编码器-解码器模型，适用于翻译、摘要、问答等任务。

3. **领域特定模型**：可以在通用预训练模型基础上，针对特定领域（如医疗、法律、金融等）进行继续预训练或微调。

4. **多语言模型**：支持训练多语言模型，只需提供相应的多语言数据集和适当的分词器。

## 模型规模与硬件要求

项目支持训练不同规模的模型，从小型研究模型到大型生产模型：

| 模型规模 | 参数量 | 最小硬件要求 | 推荐硬件配置 |
|---------|-------|------------|------------|
| 小型模型 | ~125M | 单GPU 8GB VRAM | 单GPU 16GB VRAM |
| 中型模型 | ~350M | 单GPU 16GB VRAM 或 多GPU | 多GPU，每个至少16GB VRAM |
| 大型模型 | ~760M | 多GPU，每个至少16GB VRAM | 多GPU，每个至少32GB VRAM |
| 超大模型 | >1B   | 多GPU集群 | 多GPU集群，配合模型并行技术 |

## 项目结构

```
├── data/                  # 数据目录
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后的数据
├── src/                   # 源代码
│   ├── data/              # 数据处理相关代码
│   │   ├── __init__.py
│   │   ├── preprocessing.py  # 数据预处理
│   │   └── tokenizer.py      # 分词器
│   ├── models/            # 模型相关代码
│   │   ├── __init__.py
│   │   ├── attention.py      # 注意力机制
│   │   ├── transformer.py    # Transformer模型
│   │   └── utils.py          # 模型工具函数
│   ├── training/          # 训练相关代码
│   │   ├── __init__.py
│   │   ├── trainer.py        # 训练器
│   │   └── optimizer.py      # 优化器
│   └── evaluation/        # 评估相关代码
│       ├── __init__.py
│       └── metrics.py        # 评估指标
├── configs/               # 配置文件
│   └── model_config.py    # 模型配置
├── scripts/               # 脚本文件
│   ├── train.py           # 训练脚本
│   └── evaluate.py        # 评估脚本
├── notebooks/             # Jupyter笔记本
│   └── exploration.ipynb  # 数据探索
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明
```

## 环境要求

- Python 3.8+
- CUDA 11.0+ (用于GPU加速，可选)
- 至少16GB内存（推荐32GB以上）
- 用于大规模训练的GPU（推荐至少8GB显存）

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd llms
```

2. 创建虚拟环境并安装依赖：
```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用 venv\Scripts\activate
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch >= 1.10.0
- Transformers >= 4.15.0
- Datasets >= 1.18.0
- Tensorboard >= 2.7.0
- WANDB >= 0.12.0（用于实验跟踪）

## 使用方法

### 数据准备

将原始数据放入 `data/raw/` 目录，然后运行数据预处理脚本：

```bash
python -m src.data.preprocessing
```

支持的数据格式：
- 纯文本文件（.txt）
- JSON格式（每行一个样本）
- CSV格式（包含文本列）

预处理过程包括：分词、序列切分、训练/验证/测试集划分等。

### 训练方法

#### 1. 选择模型配置

项目提供了多种预设配置，可以在 `configs/model_config.py` 中选择或自定义：

```python
# 小型模型 (~125M参数)
config = get_small_config()

# 中型模型 (~350M参数)
config = get_medium_config()

# 大型模型 (~760M参数)
config = get_large_config()
```

#### 2. 单机训练

```bash
python scripts/train.py --config configs.model_config --config_name get_small_config()
```

#### 3. 分布式训练

对于大型模型，推荐使用分布式训练：

```bash
# 使用torch.distributed.launch启动分布式训练
python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py \
    --config configs.model_config --config_name get_medium_config()
```

#### 4. 高级训练选项

##### 混合精度训练

默认启用混合精度训练以加速训练过程并减少内存使用：

```python
# 在model_config.py中设置
config.fp16 = True
```

##### 梯度检查点

对于大型模型，启用梯度检查点以减少内存使用：

```python
# 在model_config.py中设置
config.memory_optimization = {
    "gradient_checkpointing": True
}
```

##### 模型并行

对于超大型模型，启用模型并行：

```python
# 在model_config.py中设置
config.distributed_training = {
    "model_parallel": True,
    "pipeline_parallel": True
}
```

### 评估模型

```bash
python scripts/evaluate.py --model_path checkpoints/model.pt
```

## 模型架构

本项目实现了基于Transformer架构的大语言模型，包括：

- 多头自注意力机制
- 位置编码
- 前馈神经网络
- 残差连接和层归一化

## 优化技术

本项目集成了deepseek-ai的多个开源项目的优化技术，包括：

1. **FlashMLA**：高效的注意力计算机制
   - Flash Attention：内存高效的注意力计算
   - 块稀疏注意力：降低计算复杂度
   - 多查询注意力：减少内存使用
   - KV缓存：加速自回归生成

2. **DualPipe**：分布式训练优化
   - 数据并行：在多个设备上复制模型，每个设备处理不同的数据批次
   - 模型并行：将模型的不同层分布在不同的设备上
   - 流水线并行：结合数据并行和模型并行的优势

3. **EPLB**：内存优化技术
   - 梯度检查点：减少内存使用，但略微增加计算开销
   - 激活重计算：在反向传播时重新计算激活值
   - ZeRO优化：优化器状态分片

### 模型配置

可以通过修改 `configs/model_config.py` 文件来调整模型参数：

- 模型大小：支持不同参数规模的模型配置（小型约125M参数，中型约350M参数）
- 词表大小：默认使用GPT-2词表（50257个token）
- 上下文长度：最大序列长度可配置（默认1024）
- 层数和维度：可调整Transformer层数和嵌入维度
- 训练参数：学习率、批量大小、优化器参数等

## 训练技巧

- 使用梯度累积来处理大批量
- 学习率预热和衰减策略
- 混合精度训练（FP16）提高训练速度
- 模型并行和数据并行分布式训练
- 检查点保存与恢复，支持训练中断后继续
- 使用Tensorboard和WANDB进行训练监控

### 性能优化

- 使用`torch.compile`（PyTorch 2.0+）加速模型
- 内存优化技术减少显存占用
- 高效的数据加载和预处理流水线

## 训练效率优化建议

1. **数据处理优化**：使用多进程数据加载和预取，减少数据加载瓶颈

2. **批量大小调整**：根据可用GPU内存调整批量大小，必要时使用梯度累积

3. **学习率调整**：使用预热和线性衰减的学习率调度

4. **内存优化**：对于大型模型，启用梯度检查点和激活重计算

5. **分布式策略选择**：
   - 对于中小型模型：优先使用数据并行
   - 对于大型模型：考虑模型并行或流水线并行
   - 对于超大型模型：使用ZeRO优化和模型分片

## 推理优化

训练完成后，可以使用以下技术优化推理性能：

1. **KV缓存**：加速自回归生成过程

2. **量化**：使用INT8或INT4量化减少内存占用和提高推理速度

3. **批处理推理**：对多个输入进行批处理以提高吞吐量

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT系列论文](https://openai.com/research/)
- [Hugging Face Transformers库](https://github.com/huggingface/transformers)
- [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) - 高效注意力计算
- [deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe) - 分布式训练优化
- [deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB) - 内存优化技术

## 贡献指南

欢迎对本项目做出贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个Pull Request

### 代码风格

本项目使用PEP 8代码风格规范。请确保您的代码通过`flake8`检查。

## 许可证

本项目采用MIT许可证 - 详情请参见 [LICENSE](LICENSE) 文件