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
from decoder import Decoder
from PIL import Image
import os
from input_to_image_embedd import LatentNN
from assets.utils import l1_regularization, plot_losses
from sklearn.preprocessing import StandardScaler
from assets.utils import clip_contrastive_loss, ssim_loss

INPUT_DIM = 35  # Fixed input dimension
OUTPUT_DIM = 2704  # Fixed output dimension (50x50 grayscale image flattened)
TEST_SPLIT = 0.2  # Proportion of data to use for validation

def evaluate(latent, decoder, val_loader, criterion, lambda_l1=0.0):
    latent.eval()
    decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets, true_img_embeddings in val_loader:
            inputs, targets = inputs.to(latent.device), targets.to(latent.device)
            img_embeddings = latent(inputs)
            images = decoder(img_embeddings)

            clip_loss = clip_contrastive_loss(img_embeddings, true_img_embeddings.to(latent.device))
            # Reconstruction loss
            recon_loss = criterion(images, targets)
            loss = recon_loss + clip_loss
            val_loss += loss.item()
    return val_loss / len(val_loader)

def train(latent, decoder, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10, lambda_l1=1e-5):
    best_loss = float('inf')
    wait = 0
    train_losses, val_losses = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent.to(device)
    latent.device = device
    decoder.to(device)

    for epoch in range(epochs):
        latent.train()
        decoder.train()
        running_loss = 0.0
        start_time = time.time()

        for inputs, targets, img_embed in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass through latent model and decoder
            img_embeddings = latent(inputs)

            clip = clip_contrastive_loss(img_embeddings, img_embed.to(device))
            images = decoder(img_embeddings)

            # Compute reconstruction loss
            recon_loss = criterion(images, targets)
            
            # Apply L1 regularization
            l1_reg = l1_regularization(latent, lambda_l1)

            # Total loss
            loss = recon_loss + l1_reg + clip
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on validation set (without regularization)
        val_loss = evaluate(latent, decoder, val_loader, criterion, lambda_l1)
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}, Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            latent_save_path = r"models/latent_model.pth"
            decoder_save_path = r"models/decoder_model.pth"
            torch.save(latent.state_dict(), latent_save_path)
            torch.save(decoder.state_dict(), decoder_save_path)
            print(f"Latent model saved at {latent_save_path}")
            print(f"Decoder saved at {decoder_save_path}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    plot_losses(train_losses, val_losses, save_path=r"plots/loss_plot_pipeline.png")

class EmbedDataset(Dataset):
    def __init__(self, X, y, raw, transform=None):
        # Processed text embeddings (X) are converted to tensors.
        self.X = torch.tensor(X, dtype=torch.float32)
        # Image data is kept as-is and later transformed to a tensor.
        self.y = y
        # Raw text embeddings (from the .pkl) are stored as-is.
        self.raw = torch.tensor(raw, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y_img = self.y[idx]
        raw_embed = self.raw[idx]
        
        # If a transform is provided (e.g., torchvision.transforms), apply it to the image.
        if self.transform:
            y_img = self.transform(y_img)
        else:
            # Convert the grayscale image to a tensor and add a channel dimension.
            y_img = torch.tensor(y_img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            y_img = y_img / 255.0  # Normalize to [0, 1]

        # Return three elements: processed text embedding, image, and raw text embedding.
        return x, y_img, raw_embed

def prepare_dataloaders(text_embed_path, image_dir, batch_size=32, test_split=0.1):
    # Load text embeddings from the .npy file (or .pkl file if you prefer).
    text_embed = np.load(text_embed_path, allow_pickle=True).item()

    # Load raw text embeddings from the pkl file (if needed)
    try:
        with open('embeddings/image_latents.pkl', 'rb') as f:
            raw_image_embeddings = pickle.load(f)
        print("Loaded raw image embeddings from pkl file")
    except FileNotFoundError:
        print("Raw image embeddings file not found, proceeding without it")
        raw_image_embeddings = {}

    # print(raw_image_embeddings.keys())
    
    X, y, raw_embeddings = [], [], []
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for image_file in image_files:
        # Extract the key from the image filename (adjust the slicing as needed).
        image_key = os.path.splitext(image_file)[0][24:-4]
        
        # Check if this key exists in the text embeddings
        if image_key in text_embed:
            # Load image, resize to 50x50, and convert to grayscale.
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('L').resize((50, 50))
            image_array = np.array(image)
            
            # Append the processed text embedding and also keep a copy of the raw embedding.
            X.append(text_embed[image_key])
            raw_embeddings.append(raw_image_embeddings[image_key])
            y.append(image_array)
        else:
            print(f"Warning: No text embedding found for image {image_file}")

    # Standardize X (the processed text embeddings).
    scaler_x = StandardScaler()

    # Perform train/val split on all three lists simultaneously.
    X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
        X, y, raw_embeddings, test_size=TEST_SPLIT, random_state=42
    )

    X_train = scaler_x.fit_transform(X_train)
    X_val = scaler_x.transform(X_val)

    with open('scaler_x.pkl', 'wb') as f:
        pickle.dump(scaler_x, f)

    # Create datasets using the updated EmbedDataset.
    train_dataset = EmbedDataset(X_train, y_train, raw_train)
    val_dataset = EmbedDataset(X_val, y_val, raw_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = prepare_dataloaders(
        text_embed_path=r'Dataset/text_vec.npy',
        image_dir=r'Dataset\filtered_images_train'
    )
    
    print(f"Data loaders are prepared")
    print(f"Train loader size: {len(train_loader.dataset)}")

    with open(r'models/best_hyperparams.json', 'r') as f:
        hyperparams = json.load(f)

    learning_rate = hyperparams["learning_rate"]
    dropout_prob = hyperparams["dropout_prob"]
    lambda_l1 = hyperparams["lambda_l1"]
    hidden_dims = [hyperparams[f"hidden_layer_{i}"] for i in range(1,1+1)]
    epochs = 1000
    patience = 50

    # Instantiate models
    decoder = Decoder(model_weights_path=r'models\vqvae_finetuned.pth')
    latent = LatentNN(input_dim=INPUT_DIM, hidden_dims=hidden_dims, output_dim=OUTPUT_DIM, dropout_prob=dropout_prob)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(latent.parameters()) + list(decoder.parameters()), lr=learning_rate)
    train(latent, decoder, train_loader, val_loader, criterion, optimizer, epochs, patience, lambda_l1)