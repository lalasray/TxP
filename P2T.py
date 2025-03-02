import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# ==========================
# CONFIGURATION
# ==========================
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Can be "meta-llama/Llama-2-13b-chat-hf" or "google/gemma-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_LENGTH = 512  # Max token length

# ==========================
# LOAD MODEL & TOKENIZER
# ==========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto").to(DEVICE)

# 🔒 Freeze LLaMA/Gemma
for param in model.parameters():
    param.requires_grad = False

# ==========================
# PRESSURE TOKEN TRANSFORMER (Trainable)
# ==========================
class TokenTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TokenTransformer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

INPUT_DIM = 256   # PressLang pressure token dimension
OUTPUT_DIM = 4096  # Match LLM token embedding size
token_transformer = TokenTransformer(INPUT_DIM, OUTPUT_DIM).to(DEVICE)

# ==========================
# PRESSLANG DATASET
# ==========================
class PressLangDataset(Dataset):
    def __init__(self, pressure_tokens, texts):
        self.pressure_tokens = pressure_tokens  # Shape: (N, INPUT_DIM)
        self.texts = texts  # List of ground-truth text descriptions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.pressure_tokens[idx], dtype=torch.float32), self.texts[idx]

# Load PressLang dataset (Assume it's preprocessed)
pressure_data = np.load("presslang_pressure_tokens.npy")  # Shape: (N, INPUT_DIM)
text_data = open("presslang_texts.txt").read().splitlines()  # List of N texts

dataset = PressLangDataset(pressure_data, text_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================
# TRAINING SETUP
# ==========================
optimizer = torch.optim.Adam(token_transformer.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()  # Used on cosine similarity loss

sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")  # For text similarity evaluation

# ==========================
# TRAINING LOOP
# ==========================
for epoch in range(EPOCHS):
    epoch_loss = 0
    for pressure_tokens, ground_truth_texts in dataloader:
        optimizer.zero_grad()

        # Move to GPU
        pressure_tokens = pressure_tokens.to(DEVICE)

        # Transform pressure tokens
        transformed_tokens = token_transformer(pressure_tokens)

        # Tokenize ground-truth text
        text_tokens = tokenizer(ground_truth_texts, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(DEVICE)

        # Concatenate transformed pressure tokens & text
        combined_input = torch.cat((transformed_tokens, text_tokens.input_ids.float()), dim=1)

        # Forward pass (frozen LLM)
        with torch.no_grad():
            output = model.generate(input_ids=combined_input.int(), max_length=150)

        # Decode generated text
        generated_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in output]

        # Compute cosine similarity loss
        embeddings = sentence_transformer.encode(generated_texts + ground_truth_texts)
        gen_embeddings, gt_embeddings = embeddings[:BATCH_SIZE], embeddings[BATCH_SIZE:]
        
        cosine_sims = np.array([
            np.dot(gen, gt) / (np.linalg.norm(gen) * np.linalg.norm(gt))
            for gen, gt in zip(gen_embeddings, gt_embeddings)
        ])

        loss = loss_fn(torch.tensor(cosine_sims, dtype=torch.float32, requires_grad=True).to(DEVICE), 
                       torch.ones_like(torch.tensor(cosine_sims, dtype=torch.float32)).to(DEVICE))  # Target similarity = 1

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}: Loss = {epoch_loss / len(dataloader):.4f}")

