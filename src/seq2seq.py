import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import BahdanauAttention, LuongAttention

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_seq):
        embedded = self.dropout(self.embedding(input_seq))
        output, hidden = self.gru(embedded)
        # output: (batch, seq_len, hidden_size) -> Attention için gerekli
        # hidden: (1, batch, hidden_size) -> Decoder'ın başlangıç durumu
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, attn_type='bahdanau', luong_method='dot', dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.attn_type = attn_type
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Attention türünü seçme
        if attn_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_size)
        elif attn_type == 'luong':
            self.attention = LuongAttention(luong_method, hidden_size)
        else:
            raise ValueError("attn_type 'bahdanau' veya 'luong' olmalıdır.")
            
        # Luong ve Bahdanau'nun context vektörlerini bağlama (concat) ve çıktı katmanları biraz farklıdır
        # Burada basitlik ve genelleme için standart bir birleştirme yapıyoruz.
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_step, hidden, encoder_outputs):
        # input_step: (batch, 1) - Tek bir kelime (token)
        embedded = self.dropout(self.embedding(input_step)) # (batch, 1, hidden_size)
        
        # GRU adımı
        gru_out, hidden = self.gru(embedded, hidden) # gru_out: (batch, 1, hidden_size)
        
        # Attention Hesaplaması
        if self.attn_type == 'bahdanau':
            # Bahdanau genellikle önceki hidden state'i kullanır
            context_vector, attention_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        else:
            # Luong genellikle mevcut GRU çıktısını kullanır
            context_vector, attention_weights = self.attention(gru_out, encoder_outputs)
            
        # Context vektörü boyutunu ayarlama
        if len(context_vector.shape) == 2:
            context_vector = context_vector.unsqueeze(1) # (batch, 1, hidden_size)
            
        # Context ile GRU çıktısını birleştirip kelime tahminine gitme
        concat_input = torch.cat((gru_out, context_vector), dim=2) # (batch, 1, hidden_size * 2)
        output = F.log_softmax(self.out(concat_input.squeeze(1)), dim=1) # (batch, output_size)
        
        # attention_weights'i döndürmek heatmap çizimi için kritiktir!
        return output, hidden, attention_weights