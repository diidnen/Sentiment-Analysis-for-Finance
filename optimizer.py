from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

def get_optimizer(model, train_dataloader=None, num_epochs=3):
    """
    为BERT+BiLSTM模型创建优化器和调度器
    """
    # 获取所有命名参数
    param_optimizer = list(model.named_parameters())
    
    # 简化的参数分组 - 只按层类型区分学习率
    optimizer_grouped_parameters = [
        # BERT层 - 低学习率
        {'params': [p for n, p in param_optimizer if 'bert' in n],
         'lr': 2e-6},
        
        # LSTM层 - 中等学习率
        {'params': [p for n, p in param_optimizer if 'lstm' in n],
         'lr': 1e-5},
        
        # 全连接层 - 较高学习率
        {'params': [p for n, p in param_optimizer if 'fc' in n],
        'lr': 1e-5}
    ]
    
    # 计算总步数(如果提供了dataloader)
    total_steps = len(train_dataloader) * num_epochs if train_dataloader else 1000
    
    # 使用标准Adam替代BertAdam
    optimizer = Adam(optimizer_grouped_parameters)
    
    # 如果没有提供dataloader，返回None作为scheduler
    scheduler = None
    if train_dataloader:
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=[2e-6, 1e-5, 1e-5],  # 对应三个参数组的最大学习率
            steps_per_epoch=len(train_dataloader),
            epochs=num_epochs,
            pct_start=0.2,
            anneal_strategy='cos'
        )
        ''''
        为什么使用OneCycleLR？
快速收敛：初期快速增加的学习率加速训练
精细调整：后期降低的学习率允许更精细的权重调整
避免过拟合：最后阶段的低学习率减少过拟合风险
提高泛化能力：研究表明这种策略能提高模型泛化性能
这种学习率调度策略在实践中被证明非常有效，尤其适合包含预训练模型(如BERT)的复杂网络。
        
        
        '''
    
    return optimizer, scheduler