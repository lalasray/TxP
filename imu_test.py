import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from six.moves import xrange
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from vector_quant import VectorQuantizer, VectorQuantizerEMA
from enc_dec import IMUEncoder, IMUDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

'''
# Define synthetic IMU dataset
class IMUDataset(Dataset):
    def __init__(self, imu_data, seq_length=50):
        self.imu_data = imu_data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.imu_data) - self.seq_length + 1

    def __getitem__(self, idx):
        seq = self.imu_data[idx:idx + self.seq_length].T  # (channel, window)
        label = 0  # Dummy label
        return torch.tensor(seq, dtype=torch.float32), label

# Generate synthetic IMU data
num_samples = 1000
sampling_rate = 48
time = np.arange(0, num_samples / sampling_rate, 1 / sampling_rate)
acc_x = 0.5 * np.sin(2 * np.pi * 1 * time) + np.random.normal(0, 0.02, num_samples)
acc_y = 0.5 * np.cos(2 * np.pi * 1 * time) + np.random.normal(0, 0.02, num_samples)
acc_z = 9.81 + np.random.normal(0, 0.02, num_samples)
imu_data = np.stack([acc_x, acc_y, acc_z], axis=1)

seq_length = 48
batch_size = 32
imu_dataset = IMUDataset(imu_data, seq_length=seq_length)
training_loader = DataLoader(imu_dataset, batch_size=batch_size, shuffle=True)

# Calculate data variance
data_variance = np.var(imu_data / np.max(imu_data))

num_training_updates = 15000
'''

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-3

class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = IMUEncoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            
        self._decoder = IMUDecoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        # print('z.shape',z.shape)
        z = z.unsqueeze(-1)
        # print('z.shape',z.shape)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        quantized = quantized.squeeze(-1)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity,quantized
    

class Model_linear(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model_linear, self).__init__()
        
        self._encoder = IMUEncoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            
        self._decoder = IMUDecoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        z = z.unsqueeze(-1)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        quantized = quantized.squeeze(-1)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity
'''        
#model = Model(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay).to(device)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

#model.train()
#train_res_recon_error = []
#train_res_perplexity = []

#for i in xrange(num_training_updates):
#    data, _ = next(iter(training_loader))  # Only unpack data
#    data = data.to(device)
    
#    optimizer.zero_grad()
    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()
    optimizer.step()
    
    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i+1) % 100 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()

# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved.")

# Print a batch of input and reconstructed signals for comparison
data, _ = next(iter(training_loader))
data = data.to(device)
_, data_recon, _ = model(data)

data = data.cpu().numpy()
data_recon = data_recon.cpu().detach().numpy()

# Plot the first sequence in the batch
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(data[0][0], label='Original acc_x')
plt.plot(data_recon[0][0], label='Reconstructed acc_x')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data[0][1], label='Original acc_y')
plt.plot(data_recon[0][1], label='Reconstructed acc_y')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(data[0][2], label='Original acc_z')
plt.plot(data_recon[0][2], label='Reconstructed acc_z')
plt.legend()

plt.show()
'''