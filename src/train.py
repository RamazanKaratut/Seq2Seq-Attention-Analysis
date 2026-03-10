import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device):
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    target_length = target_tensor.size(1)
    batch_size = input_tensor.size(0)
    
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[0]] * batch_size, device=device)
    
    loss = 0
    for di in range(target_length):
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[:, di])
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach().unsqueeze(1)
        
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length

def evaluate_loss(dataloader, encoder, decoder, criterion, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for input_tensor, target_tensor in dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            target_length = target_tensor.size(1)
            batch_size = input_tensor.size(0)
            
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_hidden = encoder_hidden
            decoder_input = torch.tensor([[0]] * batch_size, device=device)
            
            loss = 0
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[:, di])
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().unsqueeze(1)
                
            total_loss += loss.item() / target_length
    return total_loss / len(dataloader)

def train_model(encoder, decoder, train_dataloader, val_dataloader, epochs, device, patience=5):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    os.makedirs('models', exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        total_train_loss = 0
        for input_tensor, target_tensor in train_dataloader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)
            loss = train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device)
            total_train_loss += loss
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = evaluate_loss(val_dataloader, encoder, decoder, criterion, device)
        
        print(f"Epoch {epoch+1:02}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping Mantığı
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # En iyi modeli kaydet
            torch.save(encoder.state_dict(), 'models/best_encoder.pth')
            torch.save(decoder.state_dict(), 'models/best_decoder.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n✋ Early stopping tetiklendi! En iyi Val Loss: {best_val_loss:.4f}")
                break
                
    # Test aşamasına geçmeden önce en iyi modeli geri yükle
    print("En iyi model ağırlıkları test için yükleniyor...")
    encoder.load_state_dict(torch.load('models/best_encoder.pth', weights_only=True))
    decoder.load_state_dict(torch.load('models/best_decoder.pth', weights_only=True))