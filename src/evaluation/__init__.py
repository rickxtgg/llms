"""评估模块

这个包包含了大语言模型的评估相关组件，包括各种评估指标的计算方法。
"""

from .metrics import calculate_perplexity, calculate_accuracy

__all__ = ['calculate_perplexity', 'calculate_accuracy']