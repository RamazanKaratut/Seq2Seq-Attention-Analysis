import torch
import matplotlib.pyplot as plt
import os
import pandas as pd

from src.seq2seq import EncoderRNN, AttnDecoderRNN
from src.evaluate import measure_execution_time

# 1. Ayarlar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Analiz cihazı: {device}\n")

os.makedirs("results", exist_ok=True)

# Hiperparametreler (Sadece süre ölçeceğimiz için eğitim yapmayacağız)
VOCAB_SIZE = 100
HIDDEN_SIZE = 128
SEQ_LENGTHS = [10, 50, 100, 200] # Farklı dizi uzunlukları
NUM_RUNS = 100 # Ortalamayı almak için her testi kaç kez tekrarlayacağımız

attention_types = ['dot', 'general', 'additive']
results = {attn: [] for attn in attention_types}

print("Süre ölçümleri başlıyor (Bu işlem birkaç saniye sürebilir)...")

# 2. Test Döngüsü
for seq_len in SEQ_LENGTHS:
    print(f"\n--- Dizi Uzunluğu: {seq_len} ---")
    
    # Bu uzunlukta rastgele bir girdi tensörü oluştur (Batch Size = 1)
    input_tensor = torch.randint(0, VOCAB_SIZE, (1, seq_len), device=device)
    
    # Encoder'ı oluştur (Tüm attention türleri aynı encoder çıktısını kullanacak)
    encoder = EncoderRNN(VOCAB_SIZE, HIDDEN_SIZE).to(device)
    encoder.eval()
    
    for attn_method in attention_types:
        # Seçili attention yöntemiyle Decoder'ı oluştur
        decoder = AttnDecoderRNN(HIDDEN_SIZE, VOCAB_SIZE, attn_type='luong', luong_method=attn_method).to(device)
        decoder.eval()
        
        # Süreyi ölç
        avg_time = measure_execution_time(encoder, decoder, input_tensor, device, num_runs=NUM_RUNS)
        results[attn_method].append(avg_time)
        
        print(f"{attn_method.capitalize():<10} Attention: {avg_time:.4f} ms")

# 3. Sonuçları Tabloya Dökme
df = pd.DataFrame(results, index=SEQ_LENGTHS)
df.index.name = 'Sequence Length'
print("\n--- Ödev 2 İçin README.md Tablosu ---")
print(df.to_markdown())

# 4. Performans Grafiğini Çizme ve Kaydetme
plt.figure(figsize=(10, 6))

plt.plot(SEQ_LENGTHS, results['dot'], marker='o', linestyle='-', linewidth=2, label='Dot Attention')
plt.plot(SEQ_LENGTHS, results['general'], marker='s', linestyle='--', linewidth=2, label='General Attention')
plt.plot(SEQ_LENGTHS, results['additive'], marker='^', linestyle='-.', linewidth=2, label='Additive (Concat) Attention')

plt.title('Attention Mekanizmalarının İşlem Süresi Karşılaştırması', fontsize=14)
plt.xlabel('Dizi Uzunluğu (Sequence Length)', fontsize=12)
plt.ylabel('Ortalama Süre (Milisaniye - ms)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# Grafiği results klasörüne kaydet
save_path = 'results/execution_times_plot.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"\nGrafik başarıyla kaydedildi: {save_path}")