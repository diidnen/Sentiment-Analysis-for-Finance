import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class BERTBiLSTMModel(nn.Module):
    """Combined BERT and BiLSTM model for sentiment analysis"""
    
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=256, num_classes=3, dropout_rate=0.2):
        super(BERTBiLSTMModel, self).__init__()
        
        # BERT layers
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_dropout = nn.Dropout(0.1)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=768,  # BERT hidden size
            hidden_size=hidden_size,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # Fully connected layers (修改为与BiRNN类似的设计)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),  # 使用BiLSTM首尾时间步拼接，所以是4倍hidden_size
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Sigmoid for multi-label classification
        self.sigmoid = nn.Sigmoid()
        
        # Softmax for multi-class classification
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass through the network
        Args:
            input_ids: Token ids from BERT tokenizer
            attention_mask: Mask for padding tokens
            token_type_ids: Segment ids (typically all zeros for single sentence tasks)
        """
        # BERT encoding - get sequence output and pooled output
        bert_outputs = self.bert(
            input_ids, 
            token_type_ids=token_type_ids,#对于句子对（如问答、语句相似性等），第一个句子标记为0，第二个句子标记为1
            attention_mask=attention_mask, #例子：句子"I love this"填充到长度5可能是[1,1,1,1,0]，表示最后一个是填充
            output_all_encoded_layers=False#是否输出所有层的编码结果，如果为False，只输出最后一层的编码结果
        )
        
        sequence_output = bert_outputs[0]  # [batch_size, seq_len, 768]# 用于BiLSTM的输入   
        pooled_output = bert_outputs[1]    # [batch_size, 768]  # 用于与BiLSTM隐状态拼接（就是整个句子的信息）
        
        # Apply dropout to BERT output
        sequence_output = self.bert_dropout(sequence_output)
        
        # BiLSTM processing - 只保留序列输出
        lstm_output, _ = self.lstm(sequence_output)
        # lstm_output shape: [batch_size, seq_len, hidden_size*2]
        
        # 使用类似BiRNN的首尾时间步连接方式
        # 获取第一个和最后一个时间步的输出
        first_step = lstm_output[:, 0, :]   # 第一个时间步的输出
        last_step = lstm_output[:, -1, :]   # 最后一个时间步的输出
        
        # 连接首尾时间步的输出
        # 这样可以同时捕获序列的开始和结束信息
        concat_hidden = torch.cat((first_step, last_step), dim=1)
        # concat_hidden shape: [batch_size, hidden_size*4]
        
        # 通过全连接层 进行decode
        output = self.fc(concat_hidden)
        
        # 应用sigmoid进行多标签分类
        return self.sigmoid(output)