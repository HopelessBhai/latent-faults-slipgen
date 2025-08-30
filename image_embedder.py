import os
import glob
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from assets.utils import plot_losses

# ------------------------------
# Vector Quantizer Module
# ------------------------------
class VectorQuantizer(nn.Module):
    """
    Vector Quantizer implementation for VQ-VAE.
    Maps continuous latent vectors to discrete codes from a learned codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        """
        Args:
            num_embeddings (int): Number of embeddings in the codebook.
            embedding_dim (int): Dimensionality of each embedding.
            beta (float): Commitment cost used to scale the loss.
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        # Initialize embeddings as parameters with shape (num_embeddings, embedding_dim)
        # This creates a learnable codebook of embedding vectors
        self.embeddings = nn.Parameter(torch.rand(num_embeddings, embedding_dim))

    def forward(self, inputs):
        """
        Forward pass through the Vector Quantizer.
        
        Args:
            inputs: Tensor of shape (batch, embedding_dim, height, width)
            
        Returns:
            quantized: Tensor of quantized values with same shape as input
            loss: Vector quantization loss term
        """
        # inputs shape: (batch, channels, height, width) where channels == embedding_dim
        input_shape = inputs.shape
        
        # Rearrange the input to shape (batch * height * width, embedding_dim)
        # This flattens the spatial dimensions to allow vector comparison
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous()
        flat_inputs = flat_inputs.view(-1, self.embedding_dim)

        # Compute squared Euclidean distances between the inputs and embeddings
        # Using ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y> for efficiency
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings ** 2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.embeddings.t())
        )
        # Find the nearest embedding for each input vector (argmin operation)
        encoding_indices = torch.argmin(distances, dim=1)
        # Quantize the inputs using the nearest embeddings from the codebook
        quantized = self.embeddings[encoding_indices]

        # Reshape quantized vectors back into input shape (batch, channels, height, width)
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Compute the loss terms:
        # 1. Commitment loss: ensures encoder commits to an embedding
        # 2. Codebook loss: updates the embeddings towards encoder outputs
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        loss = self.beta * commitment_loss + codebook_loss

        # Use the straight-through estimator for backpropagation
        # This copies gradients from decoder to encoder, bypassing non-differentiable quantization
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss

# ------------------------------
# VQ-VAE Model
# ------------------------------
class VQVAE(nn.Module):
    """
    Vector Quantized-Variational AutoEncoder (VQ-VAE) model.
    Consists of an encoder, vector quantizer, and decoder.
    """
    def __init__(self, latent_dim=16, num_embeddings=128, beta=0.25):
        """
        Args:
            latent_dim (int): Dimensionality of the latent space.
            num_embeddings (int): Number of vectors in the codebook.
            beta (float): Commitment cost for the vector quantizer.
        """
        super(VQVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        # Encoder: converts a (1, 50, 50) image to a latent representation (latent_dim, 13, 13)
        # Progressive downsampling through convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 50->25
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 25->13
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=1, stride=1, padding=0)  # Pointwise conv to project to latent_dim channels
        )

        # Decoder: reconstructs the image from the latent space to (1, 50, 50)
        # Progressive upsampling through transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 13->25
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 25->50
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),  # Maintain size but reduce channels to 1
            nn.Sigmoid()  # Output in [0,1] range for image pixels
        )

        # Vector quantizer layer for discrete latent representation
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, beta)

    def forward(self, x):
        """
        Forward pass through the VQ-VAE.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 50, 50)
            
        Returns:
            x_recon: Reconstructed image
            vq_loss: Vector quantization loss
        """
        # Encode input to latent representation
        z = self.encoder(x)
        # Quantize latent representation
        quantized, vq_loss = self.vq_layer(z)
        # Decode quantized representation to reconstruction
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss

# ------------------------------
# Weight Transfer (Stub)
# ------------------------------
def transfer_weights(saved_model_dir, model):
    """
    Placeholder function for transferring weights from a pre-trained model.
    
    Args:
        saved_model_dir: Directory containing saved model weights
        model: Target PyTorch model
    """
    # Transferring weights from a TensorFlow SavedModel to a PyTorch model is not supported.
    print("transfer_weights: Not implemented in PyTorch version. Skipping.")

# ------------------------------
# Dataset and Data Loading
# ------------------------------
class ImageDataset(Dataset):
    """
    Custom dataset for loading and preprocessing images.
    """
    def __init__(self, directory, transform=None, return_filenames=True):
        """
        Args:
            directory (str): Directory containing image files.
            transform: Optional transform to apply to the images.
            return_filenames (bool): Whether to return file paths with images.
        """
        self.directory = directory
        self.file_list = glob.glob(os.path.join(directory, "*.png"))
        self.return_filenames = return_filenames
        if transform is None:
            # Default transformations: grayscale, resize, convert to tensor
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((50, 50)),
                transforms.ToTensor()  # Converts image to [0,1] range
            ])
        else:
            self.transform = transform

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Retrieves an image by index.
        
        Args:
            idx (int): Index of the image to retrieve.
            
        Returns:
            tuple: (image, filepath) or just image depending on return_filenames
        """
        filepath = self.file_list[idx]
        image = Image.open(filepath)
        image = self.transform(image)
        if self.return_filenames:
            return image, filepath
        else:
            return image

# ------------------------------
# Training (Fine-Tuning) Routine
# ------------------------------
def fine_tune_vqvae(model, dataset, epochs=10, lr=1e-4, save_path="models/vqvae_finetuned.pth", 
                    val_split=0.2, batch_size=16, device="cpu", plot_loss_fn=None):
    """
    Fine-tune a VQ-VAE model on the given dataset.
    
    Args:
        model: The VQ-VAE model to train.
        dataset: Dataset containing training data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimizer.
        save_path (str): Path to save the trained model.
        val_split (float): Fraction of data to use for validation.
        batch_size (int): Batch size for training.
        device: Device to train on (cuda/cpu).
        plot_loss_fn: Optional function to plot training losses.
    """
    # Split the dataset into training and validation sets
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for batched processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.MSELoss()  # Mean squared error loss for reconstruction

    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20

    # Move model to the specified device
    model.to(device)

    # Lists to track loss progression
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [Training]", leave=False)
        for images, _ in loop:
            images = images.to(device)
            optimizer.zero_grad()  # Zero gradients before backward pass
            reconstructions, vq_loss = model(images)  # Forward pass
            recon_loss = criterion(reconstructions, images)  # Calculate reconstruction loss
            loss = recon_loss + vq_loss  # Total loss includes VQ loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            train_loss += loss.item() * images.size(0)  # Accumulate batch loss
            loop.set_postfix(loss=loss.item())  # Update progress bar
        train_loss /= train_size  # Calculate average loss
        train_losses.append(train_loss)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        loop = tqdm(val_loader, desc=f"Epoch {epoch} [Validation]", leave=False)
        with torch.no_grad():  # No need to track gradients during validation
            for images, _ in loop:
                images = images.to(device)
                reconstructions, vq_loss = model(images)
                recon_loss = criterion(reconstructions, images)
                loss = recon_loss + vq_loss
                val_loss += loss.item() * images.size(0)
                loop.set_postfix(loss=loss.item())
        val_loss /= val_size
        val_losses.append(val_loss)

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            patience_counter = 0
        else:
            # Early stopping logic
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    print(f"VQ-VAE fine-tuning complete; weights saved to '{save_path}'.")

    # Plot training and validation losses if plot function provided
    if plot_loss_fn:
        plot_losses(train_losses, val_losses, save_path=r'plots/loss_plot_image_embedder.png')

# ------------------------------
# Latent Embedding Extraction
# ------------------------------
def extract_latents(model, dataset, save_path=r"embeddings/image_latents.pkl", batch_size=16, device="cpu"):
    """
    Extract and save latent embeddings for all images in the dataset using the trained encoder.
    
    Args:
        model: Trained VQ-VAE model.
        dataset: Dataset containing images.
        save_path (str): Path to save the extracted embeddings.
        batch_size (int): Batch size for processing.
        device: Device to run inference on.
    """
    model.eval()  # Set model to evaluation mode
    latent_dict = {}  # Dictionary to store embeddings
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Process all images without computing gradients
    with torch.no_grad():
        for images, filepaths in dataloader:
            images = images.to(device)
            # Get latent representations from the encoder
            latents = model.encoder(images)
            # Flatten the latent representations for storage
            latents_flat = latents.view(latents.size(0), -1).cpu().numpy()
            for i, fp in enumerate(filepaths):
                # Extract a unique key from the filename
                key = os.path.splitext(os.path.basename(fp))[0][24:-4]
                latent_dict[key] = latents_flat[i]
    
    # Create directory if it doesn't exist and save embeddings
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(latent_dict, f)
    print(f"Saved latent embeddings to '{save_path}'.")

# ------------------------------
# Main Routine
# ------------------------------
def main():
    """
    Main function to execute the training pipeline.
    """
    
    # Set the directory containing training images
    heatmap_dir = r"Dataset/filtered_images_train"

    # Create dataset
    dataset = ImageDataset(heatmap_dir, return_filenames=True)
    
    # Determine device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the VQ-VAE model
    vqvae_model = VQVAE(latent_dim=16, num_embeddings=128).to(device)
    
    # Weight transfer is not supported in this PyTorch version
    # transfer_weights(r"models/keras_vqvae", vqvae_model)
    
    # Define paths for saving model weights and embeddings
    save_path_weights = r"models/vqvae_finetuned.pth"
    save_path_embed = r"embeddings/image_latents.pkl"
    
    # Fine-tune the model
    fine_tune_vqvae(vqvae_model, dataset, epochs=1000, lr=1e-4, save_path=save_path_weights, device=device, plot_loss_fn=True)
    # Extract latent embeddings
    extract_latents(vqvae_model, dataset, save_path=save_path_embed, device=device)

# Entry point of the script
if __name__ == "__main__":
    main()