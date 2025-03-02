import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# --------------------
# CONFIGURATIONS
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_PRESSURE_LEN = 64  # Max pressure token sequence length
BATCH_SIZE = 16
LR = 3e-5
EPOCHS = 5
EMBEDDING_DIM = 512  # CLIP text embedding size

# --------------------
# LOAD CLIP MODEL
# --------------------
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()  # Freeze CLIP during training

# --------------------
# DATASET CLASS
# --------------------
class PressLangDataset(Dataset):
    """Dataset for text-to-pressure mapping using CLIP embeddings"""

    def __init__(self, text_list, pressure_tokens):
        self.text_list = text_list
        self.pressure_tokens = pressure_tokens

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        pressure_seq = self.pressure_tokens[idx]

        # Convert text to CLIP embeddings
        with torch.no_grad():
            text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            text_emb = clip_model.get_text_features(**{k: v.to(DEVICE) for k, v in text_inputs.items()})
        
        text_emb = text_emb.squeeze(0)  # Remove batch dim

        # Convert pressure tokens to tensor
        pressure_seq = torch.tensor(pressure_seq, dtype=torch.long)

        return text_emb, pressure_seq

# --------------------
# LOAD DATA
# --------------------
def load_data():
    """Load dataset containing (text, pressure token sequence) pairs"""
    text_data = np.load("text_descriptions.npy")  # Example file containing text descriptions
    pressure_data = np.load("pressure_tokens.npy")  # Corresponding pressure token sequences

    dataset = PressLangDataset(text_data, pressure_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# --------------------
# MODEL DEFINITION
# --------------------
class PressTokenTransformer(nn.Module):
    """Autoregressive Transformer for predicting pressure tokens from CLIP embeddings"""

    def __init__(self, vocab_size, num_layers=6, num_heads=8, hidden_dim=512):
        super().__init__()

        self.linear_proj = nn.Linear(EMBEDDING_DIM, hidden_dim)  # Map CLIP embeddings to transformer input space
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(hidden_dim, vocab_size)  # Predict next token

    def forward(self, clip_text_emb, pressure_tokens):
        """
        Forward pass for training with teacher forcing.
        clip_text_emb: (batch, embedding_dim)
        pressure_tokens: (batch, pressure_seq_len)
        """
        text_emb = self.linear_proj(clip_text_emb).unsqueeze(1)  # Map CLIP embeddings to hidden_dim
        pressure_emb = self.embedding(pressure_tokens)

        # Transformer forward
        transformer_out = self.transformer(
            src=text_emb, 
            tgt=pressure_emb
        )
        output = self.fc_out(transformer_out)  # Predict next pressure token

        return output

# --------------------
# TRAINING FUNCTION
# --------------------
def train_model():
    train_loader, val_loader = load_data()
    vocab_size = 1028  # Example vocab size for pressure tokens
    model = PressTokenTransformer(vocab_size).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for clip_text_emb, pressure_tokens in train_loader:
            clip_text_emb, pressure_tokens = clip_text_emb.to(DEVICE), pressure_tokens.to(DEVICE)

            optimizer.zero_grad()
            output = model(clip_text_emb, pressure_tokens[:, :-1])  # Teacher forcing
            loss = criterion(output.view(-1, vocab_size), pressure_tokens[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        validate_model(model, val_loader)

    torch.save(model.state_dict(), "pressure_transformer.pth")
    print("Model saved.")

# --------------------
# VALIDATION FUNCTION
# --------------------
def validate_model(model, val_loader):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for clip_text_emb, pressure_tokens in val_loader:
            clip_text_emb, pressure_tokens = clip_text_emb.to(DEVICE), pressure_tokens.to(DEVICE)
            output = model(clip_text_emb, pressure_tokens[:, :-1])
            loss = criterion(output.view(-1, 1028), pressure_tokens[:, 1:].reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

# --------------------
# GENERATION FUNCTION
# --------------------
def generate_pressure_sequence(text, model, max_len=MAX_PRESSURE_LEN):
    model.eval()
    
    with torch.no_grad():
        text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        clip_text_emb = clip_model.get_text_features(**{k: v.to(DEVICE) for k, v in text_inputs.items()}).squeeze(0)
    
    clip_text_emb = clip_text_emb.unsqueeze(0)  # Add batch dim
    generated = [0]  # Start token (assuming 0 is <SOS>)

    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor(generated, dtype=torch.long, device=DEVICE).unsqueeze(0)
            output = model(clip_text_emb, input_tensor)
            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            generated.append(next_token)
            if next_token == 1:  # Assuming 1 is <EOS>
                break

    return generated[1:]  # Remove start token

# --------------------
# MAIN EXECUTION
# --------------------
if __name__ == "__main__":
    train_model()
