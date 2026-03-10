import torch
import matplotlib.pyplot as plt
import seaborn as sns
import time

def evaluate_and_get_attention(input_tensor, encoder, decoder, max_length, device):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_hidden = encoder_hidden
        
        decoder_input = torch.tensor([[0]], device=device) # SOS token
        
        decoded_words = []
        input_length = input_tensor.size(1)
        
        # ÇÖZÜM: Matrisin sütun sayısını doğrudan girdi uzunluğuna (input_length) eşitliyoruz
        decoder_attentions = torch.zeros(max_length, input_length)
        
        for di in range(max_length):
            decoder_output, decoder_hidden, attention_weights = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            attn_flat = attention_weights.view(-1)
            decoder_attentions[di, :attn_flat.size(0)] = attn_flat.cpu()
            
            topv, topi = decoder_output.topk(1)
            if topi.item() == 1: # EOS token
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(str(topi.item())) 
                
            decoder_input = topi.squeeze().detach().unsqueeze(0).unsqueeze(0)
            
        return decoded_words, decoder_attentions[:di + 1]

def plot_attention_heatmap(input_sentence, translated_words, attentions, save_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    sns.heatmap(attentions.numpy(), cmap='viridis', ax=ax, cbar=True)
    
    # ÇÖZÜM: Verimizde girişte <EOS> olmadığı için sadece kelimeleri etiketliyoruz
    ax.set_xticklabels(input_sentence.split(), rotation=90)
    ax.set_yticklabels(translated_words, rotation=0)
    
    plt.title("Attention Weights Heatmap")
    plt.xlabel("Kaynak Dizi")
    plt.ylabel("Hedef Dizi")
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()

def measure_execution_time(encoder, decoder, input_tensor, device, num_runs=100):
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[0]] * input_tensor.size(0), device=device)
    
    for _ in range(10):
        _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
        
    start_time = time.time()
    for _ in range(num_runs):
        _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) / num_runs) * 1000
    return avg_time_ms