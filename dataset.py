import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch.nn as nn

class FinancialSentimentDataset(Dataset):
    """Dataset for financial text data with sentiment labels"""
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        """
        Initialize the dataset
        Args:
            texts: List of text strings
            labels: Optional list of labels (for training/validation)
            tokenizer: BERT tokenizer
            max_len: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 使用旧版BertTokenizer的方法替代encode_plus
        tokens = self.tokenizer.tokenize('[CLS] ' + text + ' [SEP]')
        
        # 截断或填充到最大长度
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            padding = ['[PAD]'] * (self.max_len - len(tokens))
            tokens = tokens + padding
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        
        # 创建注意力掩码
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[input_ids != 0] = 1  # 非PAD位置为1
        
        # 创建标记类型ID（单句全0）
        token_type_ids = torch.zeros_like(input_ids)
        
        if self.labels is not None:
            # 对于分类任务使用long类型
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return {
                'input_ids': input_ids, 
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'label': label
            }
        else:
            return {
                'input_ids': input_ids, 
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

def prepare_dataloaders(data_file, batch_size=16, max_len=256, train_ratio=0.8, val_ratio=0.1):
    """
    从带标签的文本文件准备数据加载器
    
    Args:
        data_file: 带标签的文本文件路径（格式：句子@标签）
        batch_size: 批处理大小
        max_len: 最大序列长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    # 读取文件
    texts = []
    labels = []
    
    with open(data_file, 'r', encoding='latin1') as f:  # 尝试使用latin1编码
        for line in f:
            line = line.strip()
            if line:
                # 按@分割句子和标签
                parts = line.split('@')
                if len(parts) == 2:
                    sentence, label = parts
                    texts.append(sentence.strip())
                    labels.append(label.strip())
    
    # 标签映射
    label_map = {'negative':0, 'neutral':1, 'positive':2}
    
    # 转换标签为数字
    numeric_labels = [label_map[label] for label in labels]
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 划分数据集
    data_size = len(texts)
    indices = list(range(data_size))
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * data_size)
    val_size = int(val_ratio * data_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 创建数据集
    train_dataset = FinancialSentimentDataset(
        texts=[texts[i] for i in train_indices],
        labels=[numeric_labels[i] for i in train_indices],
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    val_dataset = FinancialSentimentDataset(
        texts=[texts[i] for i in val_indices],
        labels=[numeric_labels[i] for i in val_indices],
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    test_dataset = FinancialSentimentDataset(
        texts=[texts[i] for i in test_indices],
        labels=[numeric_labels[i] for i in test_indices],
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # 创建训练集的数据加载器
    train_loader = DataLoader(
        train_dataset,          # 训练数据集对象，包含所有训练样本
        batch_size=batch_size,  # 每批处理的样本数，影响训练速度和内存使用
        shuffle=True,           # 在每个周期随机打乱数据，提高模型泛化能力
        num_workers=4,          # 用于并行加载数据的子进程数，加速数据加载
        pin_memory=True,        # 将数据放入CUDA固定内存，加速GPU训练数据传输
        drop_last=True          # 丢弃最后一个不完整的批次，避免批归一化问题
    )
    
    # 创建验证集的数据加载器
    val_loader = DataLoader(
        val_dataset,            # 验证数据集对象
        batch_size=batch_size*2, # 验证时使用更大的批量，因为不需要计算梯度可节省内存
        shuffle=False,          # 不打乱验证数据，保持顺序一致便于分析
        num_workers=4,          # 并行加载数据的进程数
        pin_memory=True         # 使用固定内存提高GPU传输效率
    )                           # 默认drop_last=False，确保验证所有样本
    
    # 创建测试集的数据加载器
    test_loader = DataLoader(
        test_dataset,           # 测试数据集对象
        batch_size=batch_size*2, # 测试时可使用更大批量，加速预测
        shuffle=False,          # 不打乱测试数据，保持原始顺序
        num_workers=4,          # 并行加载数据的进程数
        pin_memory=True         # 使用固定内存提高GPU传输效率
    )                           # 默认drop_last=False，确保测试所有样本
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

