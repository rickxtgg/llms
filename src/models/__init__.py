"""模型模块

这个包包含了大语言模型的核心架构实现，包括注意力机制、Transformer结构等组件。
"""

from .transformer import GPT

__all__ = ['GPT']