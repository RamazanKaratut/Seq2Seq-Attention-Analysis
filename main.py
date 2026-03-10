import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import os

from src.seq2seq import EncoderRNN, AttnDecoderRNN
from src.train import train_model
from src.evaluate import evaluate_and_get_attention, plot_attention_heatmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

os.makedirs("results", exist_ok=True)

INPUT_VOCAB_SIZE = 10 
OUTPUT_VOCAB_SIZE = 10
HIDDEN_SIZE = 128
MAX_LENGTH = 7
BATCH_SIZE = 16
EPOCHS = 100 # Early stopping olduğu için yüksek tutabiliriz
PATIENCE = 5 # 5 epoch boyunca val loss düşmezse eğitimi durdur

def generate_dummy_data(num_samples, seq_len):
    inputs = []
    targets = []
    for _ in range(num_samples):
        seq = [random.randint(2, 9) for _ in range(seq_len)]
        inputs.append(seq)
        targets.append(seq[::-1]) 
    return torch.tensor(inputs), torch.tensor(targets)

# Toplam 600 veri üretip %80 Train, %20 Val olarak bölüyoruz
x_data, y_data = generate_dummy_data(600, MAX_LENGTH - 1)
train_dataset = TensorDataset(x_data[:480], y_data[:480])
val_dataset = TensorDataset(x_data[480:], y_data[480:])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Modeller oluşturuluyor...")
encoder = EncoderRNN(INPUT_VOCAB_SIZE, HIDDEN_SIZE).to(device)
decoder = AttnDecoderRNN(HIDDEN_SIZE, OUTPUT_VOCAB_SIZE, attn_type='bahdanau').to(device)

print("Eğitim başlıyor...")
train_model(encoder, decoder, train_dataloader, val_dataloader, EPOCHS, device, PATIENCE)
print("Eğitim tamamlandı!")

print("\n--- Çeviri ve Attention Heatmap Çıkarımı ---")
test_sequences = [
    [2, 3, 4, 5, 6],
    [8, 7, 2, 9, 3],
    [5, 5, 2, 4, 8]
]

for i, test_seq in enumerate(test_sequences):
    input_tensor = torch.tensor([test_seq], device=device)
    input_sentence = " ".join([str(val) for val in test_seq])
    
    decoded_words, attentions = evaluate_and_get_attention(input_tensor, encoder, decoder, MAX_LENGTH, device)
    
    print(f"\nTest {i+1}:")
    print(f"Kaynak: {input_sentence}")
    print(f"Hedef: {' '.join(decoded_words)}")
    
    save_path = f"results/heatmap_sentence_{i+1}.png"
    attentions_matrix = attentions.squeeze().detach() 
    
    plot_attention_heatmap(input_sentence, decoded_words, attentions_matrix, save_path)

print("\nTüm işlemler bitti. Heatmap görsellerini 'results' klasöründe bulabilirsin!")