import numpy as np
import torch
import matplotlib.pyplot as plt
from imu_test import Model as VQ_VAE_Model
from train_ae_unchanged import set_seed, resize_data, CONFIG  # Reusing from train script

# Set random seed
set_seed(CONFIG["random_seed"])


# CONFIG["model_save_path"] = "/home/dfki/dee/VQKANClassifier-main/vqvaemodel_wisdom_1.pth"
# Load dataset
CONFIG["model_save_path"] = "/home/dfki/dee/VQKANClassifier-main/vqvaemodel_wisdom_1.pth"
root_path = "/home/dfki/dee/VQKANClassifier-main/Datasets/"
dataset = "wisdm_30hz_clean"
X = np.load(f"{root_path}/{dataset}/X.npy")
Y = np.load(f"{root_path}/{dataset}/Y.npy")

dict_1 = {}

og =  np.transpose(X, (0, 2, 1)) 
print(og.shape)



device = "cuda" if torch.cuda.is_available() else "cpu"
model = VQ_VAE_Model(**CONFIG["model_params"]).to(device)
model_path = CONFIG["model_save_path"].format(fold=1)  # Change fold if needed
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

final_reconstructed_batches = []
for j in range(X.shape[0]):
    reconstructed_datas = []
    total_time = X.shape[1]
    num_windows = total_time // CONFIG["input_length"]
    end_Col = num_windows*CONFIG["input_length"]
    num_windows+=1 if end_Col != total_time else num_windows
    label = Y[j]
    # print('Y:',label)

    for i in range(num_windows):
        if i < num_windows-1:
            
    
            input_data = X[j, i * CONFIG["input_length"]:(i + 1) * CONFIG["input_length"], :]
            input_data = np.expand_dims(input_data, axis=0)
            # print(f'input daza the layer{i} batch{j}',input_data.shape)

            input_data = torch.tensor(input_data, dtype=torch.float32)
            input_data = np.transpose(input_data, (0, 2, 1)) 
            with torch.no_grad():
                vq_loss, reconstructed_data, _ = model(input_data.to(device))
                reconstructed_data = reconstructed_data.cpu().numpy()

            reconstructed_datas.append(reconstructed_data)

        else:
            input_data = X[j,end_Col:,:]
            input_data = np.expand_dims(input_data, axis=0) 
     
            dee = input_data.shape[1]
            pad_value = CONFIG["input_length"] - input_data.shape[1]
            input_data = np.pad(input_data, ((0, 0), (0, pad_value), (0, 0)), mode='constant', constant_values=0)
            # print(f'input daza the layer{i} batch{j}',input_data.shape)
            input_data = torch.tensor(input_data, dtype=torch.float32)
            input_data = np.transpose(input_data, (0, 2, 1)) 
            with torch.no_grad():
                vq_loss, reconstructed_data, _ = model(input_data.to(device))
                reconstructed_data = reconstructed_data.cpu().numpy()
            
            reconstructed_datas.append(reconstructed_data[:, :, :dee])

        
    stacked_data_time = np.concatenate(reconstructed_datas, axis=-1)
    # print(stacked_data_time.shape)

    # final_reconstructed_batches.append(stacked_data_time.squeeze(0))


#     dict_1 =  {
#         'tokens': stacked_data_time,
#         'label' : label
#     }


# #extra 
# final_output = np.stack(final_reconstructed_batches, axis=0)  # Shape (batch, 3, time=300)

# print("Final stacked output shape:", final_output.shape)

# print(dict_1['tokens'].shape)
    batch_dict = {
            "data": stacked_data_time,  # Remove singleton dimension
            "label": label
        }
        
    final_reconstructed_batches.append(batch_dict)

# Example: Accessing the data
# print("Final reconstructed batches dictionary:")
for i, batch in enumerate(final_reconstructed_batches):
    print(f"Batch {i}: Data Shape = {batch['data'].shape}, Label = {batch['label']}")


torch.save(final_reconstructed_batches, "dictionary.pt")