import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from threat_c_dga.model import DGALSTM
from threat_c_dga.tokenizer import domain_to_tensor

class DomainDataset(Dataset):
    def __init__(self, df):
        self.domains = df['domain'].values
        self.labels = df['is_dga'].values.astype('float32')

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        tensor = domain_to_tensor(str(self.domains[idx]))
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return tensor, label

def load_dataset(csv_path):
    # Fallback to synthetic if real dataset (UMUDGA/Majestic Million) is missing
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        sample_data = {
            'domain': [
                'google.com', 'wikipedia.org', 'github.com', 'microsoft.com',
                'amazon.in', 'youtube.com', 'reddit.com', 'stackoverflow.com',
                'xeogrhxquuubt.ru', 'pqzmwnxyeuq823.cc', '789xczv781.biz',
                'kjashdf8723kjahsd.info', 'vbnmzxcqwepoiuy.org', '18237912hksad.net'
            ],
            'is_dga': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(csv_path, index=False)
        return df
    return pd.read_csv(csv_path).dropna(subset=['domain', 'is_dga'])

def train_dga_model():
    dataset_path = os.path.join("datasets", "dga_domains.csv")
    model_save_path = os.path.join("saved_models", "dga_lstm.pth")
    
    df = load_dataset(dataset_path)
    dataset = DomainDataset(df)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = DGALSTM()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("Starting BiLSTM Attention training on domain sequences...")
    model.train()
    epochs = 40
    for epoch in range(epochs):
        epoch_loss = 0.0
        for sequences, labels in dataloader:
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss / len(dataloader):.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"PyTorch BiLSTM model saved successfully to {model_save_path}")

if __name__ == "__main__":
    train_dga_model()