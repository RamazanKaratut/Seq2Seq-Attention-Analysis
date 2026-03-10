import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mekanizması
    Skor Fonksiyonu: v^T * tanh(W_1 * h_decoder + W_2 * h_encoder)
    """
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, 1, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        
        # Decoder'ın gizli durumunu encoder çıktıları boyutuna genişletiyoruz
        hidden_with_time_axis = hidden.unsqueeze(1)
        
        # Attention skoru hesaplama
        score = self.V(torch.tanh(self.W1(hidden_with_time_axis) + self.W2(encoder_outputs)))
        
        # Ağırlıkları (0 ile 1 arasında) bulmak için softmax uyguluyoruz
        attention_weights = F.softmax(score, dim=1)
        
        # Context vektörünü hesaplama: ağırlıklar * encoder çıktıları
        context_vector = attention_weights * encoder_outputs
        context_vector = torch.sum(context_vector, dim=1)
        
        return context_vector, attention_weights


class LuongAttention(nn.Module):
    """
    Luong Attention Mekanizması
    Desteklenen Skor Fonksiyonları: 'dot', 'general', 'additive' (concat)
    """
    def __init__(self, method, hidden_size):
        super(LuongAttention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method not in ['dot', 'general', 'additive']:
            raise ValueError(self.method, "Geçerli bir attention metodu değil.")
            
        if self.method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.method == 'additive':
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Parameter(torch.rand(hidden_size))

    def dot_score(self, hidden, encoder_outputs):
        # h_t^T * h_s
        return torch.sum(hidden * encoder_outputs, dim=2)

    def general_score(self, hidden, encoder_outputs):
        # h_t^T * W * h_s
        energy = self.W(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def additive_score(self, hidden, encoder_outputs):
        # v^T * tanh(W * [h_t; h_s])
        seq_len = encoder_outputs.size(1)
        hidden = hidden.repeat(1, seq_len, 1) # [batch, seq_len, hidden_size]
        energy = torch.tanh(self.W(torch.cat((hidden, encoder_outputs), dim=2))) # [batch, seq_len, hidden_size]
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, 1, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        
        if self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'additive':
            attn_energies = self.additive_score(hidden, encoder_outputs)
            
        # Boyutları eşitleyip softmax uyguluyoruz
        attention_weights = F.softmax(attn_energies, dim=1).unsqueeze(2) # (batch_size, seq_len, 1)
        
        # Context vektörü
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs) # (batch_size, 1, hidden_size)
        context_vector = context_vector.squeeze(1)
        
        return context_vector, attention_weights