import pandas as pd
import numpy as np
import torch
import pickle
import time
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from assets.utils import clip_contrastive_loss, plot_losses, l1_regularization

def infer_input_dim_from_file(npy_path: str) -> int:
    """
    Infer input dimension from a numpy .npy file that stores a dict of key -> vector.
    Returns the flattened length of the first vector found.
    """
    data = np.load(npy_path, allow_pickle=True).item()
    try:
        first_key = next(iter(data))
    except StopIteration:
        raise ValueError(f"No entries found in {npy_path}")
    arr = np.asarray(data[first_key]).reshape(-1)
    return int(arr.shape[0])

class LatentNN(nn.Module):
    def __init__(self, input_dim=7, hidden_dims=[2048, 1024], output_dim=2704, dropout_prob=0.3):
        """
        Modified model to accept hyperparameters.
        """
        super(LatentNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

    def predict(self, X_input):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            X_input = torch.tensor(X_input, dtype=torch.float32).to(device)
            if len(X_input.shape) == 1:
                X_input = X_input.unsqueeze(0)
            output = self.fc(X_input)
        return output.cpu().numpy()

def evaluate(model, val_loader, criterion, lambda_l1=0.0):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # loss += l1_regularization(model, lambda_l1)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10, lambda_l1=1e-5, show_plot=True):
    best_loss = float('inf')
    wait = 0
    train_losses, val_losses = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.device = device

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs.shape, targets.shape)
            # break
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            # break
            loss = criterion(outputs, targets)
            loss += l1_regularization(model, lambda_l1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = evaluate(model, val_loader, criterion, lambda_l1)
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}, Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            save_path = r"models/input_to_image_embedd.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    if show_plot:
        plot_losses(train_losses, val_losses)

class EmbedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_dataloaders(text_embed_path, image_embed_path, batch_size=32, test_split=0.1):
    # Load embeddings
    text_embed = np.load(text_embed_path, allow_pickle=True).item()
    with open(image_embed_path, 'rb') as f:
        image_embed = pickle.load(f)
        
    # Match embeddings by keys
    text_names = list(text_embed.keys())
    X, y = [], []
    for name in text_names:
        if name in image_embed:
            X.append(text_embed[name])
            y.append(image_embed[name])

    X = np.array(X)
    y = np.array(y)

    # Standardize X
    scaler_x = StandardScaler()

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_split, random_state=42)

    X_train = scaler_x.fit_transform(X_train)
    X_val = scaler_x.transform(X_val)

    with open('scaler_x.pkl', 'wb') as f:
        pickle.dump(scaler_x, f)

    # Dataset and loaders
    train_dataset = EmbedDataset(X_train, y_train)
    val_dataset = EmbedDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage with default hyperparameters
    train_loader, val_loader = prepare_dataloaders(
        text_embed_path=r'Dataset/text_vec.npy',
        image_embed_path=r'embeddings/image_latents.pkl'
    )

    print("Train loader size:", len(train_loader.dataset))
    print("Validation loader size:", len(val_loader.dataset))


    # Extract hyperparameters
    with open(r'models/best_hyperparams.json', 'r') as f:
        hyperparams = json.load(f)

    # Extract hyperparameters
    learning_rate = hyperparams["learning_rate"]
    dropout_prob = hyperparams["dropout_prob"]
    lambda_l1 = hyperparams["lambda_l1"]
    hidden_dims = [hyperparams[f"layer_{i}_size"] for i in range(hyperparams["num_layers"])]
    input_dim = infer_input_dim_from_file(r'Dataset/text_vec.npy')
    output_dim = 2704  # Fixed output_dim to match VQ-VAE latent space (16x13x13)
    epochs = 400  # Default value
    patience = 20  # Default value

    # Update model and training parameters
    model = LatentNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout_prob=dropout_prob)
    # criterion = torch.nn.MSELoss()
    criterion = clip_contrastive_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, val_loader, criterion, optimizer, epochs, patience, lambda_l1, show_plot=True)