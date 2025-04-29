import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from image_embedder import VQVAE  # Assumes PyTorch model is here

class Decoder(nn.Module):
    def __init__(self, model_weights_path, device="cpu"):
        super(Decoder, self).__init__()
        self.device = torch.device(device)
        self.model = VQVAE(latent_dim=16, num_embeddings=128)
        if model_weights_path is not None:
            state = torch.load(model_weights_path, map_location=self.device)
            new_state = {}
            for k, v in state.items():
                # Remove the "model." prefix if present.
                if k.startswith("model."):
                    new_key = k[len("model."):]
                else:
                    new_key = k
                new_state[new_key] = v
            self.model.load_state_dict(new_state)

    def forward(self, embedding):
        """
        Expects `embedding` of shape [B, 2704] (flattened latent vector)
        and reshapes it to [B, 16, 13, 13] before passing it to the decoder.
        """
        B = embedding.size(0)
        latent = embedding.view(B, 16, 13, 13)
        decoded = self.model.decoder(latent)
        return decoded

    def visualize_prediction(self, decoded_image, true_image_path=None, save_path=None):
        """
        Visualize the decoded image vs. ground truth.
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        # Ensure tensor is detached and moved to CPU for visualization.
        # print(decoded_image.shape)
        plt.imshow(decoded_image[0, 0].detach().cpu().numpy(), cmap='seismic_r')
        plt.title("Decoded Image")
        plt.axis("off")
        if true_image_path:
            true_image = Image.open(true_image_path).convert('L').resize((50,50))
            plt.subplot(1, 2, 2)
            plt.imshow(np.array(true_image), cmap='seismic_r')  # Using viridis colormap instead of gray
            plt.title("Ground Truth")
            plt.axis("off")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

# Example usage (for testing)
if __name__ == "__main__":
    embeddings_path = r"embeddings/image_latents.pkl"
    model_weights_path = r"models/vqvae_finetuned.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = Decoder(model_weights_path=model_weights_path, device=device)
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    keys = list(embeddings.keys())
    for key in keys:
        embedding = embeddings[key]
        # Convert to tensor and add a batch dimension
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)  # shape: [1, 784]
        reconstructed_image = decoder(embedding_tensor)  # Use forward() directly
        actual_image_path = rf"Dataset\filtered_images_train\interpolated_slip_image_{key}.fsp.png"
        save_path = rf"Dataset/reconstructed_images/reconstructed_image_{key}.png"
        decoder.visualize_prediction(reconstructed_image, true_image_path=actual_image_path, save_path=save_path)