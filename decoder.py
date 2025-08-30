import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from image_embedder import VQVAE  # Assumes PyTorch model is here
from assets.utils import save_metrics_for_image, pixels_to_slip

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

    def visualize_prediction(self, decoded_image, true_image_path=None, save_path=None, dz=None, image_name=None):
        """
        Visualize the decoded image vs. ground truth.
        """
        # Prepare predicted and ground-truth arrays
        pred = decoded_image[0, 0].detach().cpu().numpy()
        # Normalize predicted to [0, 1] if necessary
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = np.clip(pred, 0.0, 1.0)

        gt_array = None
        if true_image_path:
            true_image = Image.open(true_image_path).convert('L').resize((50, 50))
            gt_array = np.asarray(true_image).astype(np.float32)
            if gt_array.max() > 1.0:
                gt_array = gt_array / 255.0

        # Convert to slip scale if dz provided
        if dz is not None:
            pred_slip = pixels_to_slip(pred, dz, image_name=image_name, plot=False)
            gt_slip = pixels_to_slip(gt_array, dz, image_name=image_name, plot=False) if gt_array is not None else None
            
            # pred_slip = pred
            # gt_slip = gt_array
            
            # Shared color scale
            vmin = np.min(pred_slip if gt_slip is None else np.minimum(pred_slip, gt_slip))
            vmax = np.max(pred_slip if gt_slip is None else np.maximum(pred_slip, gt_slip))

            fig, axes = plt.subplots(1, 2 if gt_slip is not None else 1, figsize=(12, 6))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            im0 = axes[0].imshow(pred_slip, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
            axes[0].set_title("Predicted (slip)")
            axes[0].axis("off")

            if gt_slip is not None and len(axes) > 1:
                axes[1].imshow(gt_slip, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
                axes[1].set_title("Ground Truth (slip)")
                axes[1].axis("off")

            # Shared colorbar
            fig.colorbar(im0, ax=axes, fraction=0.046, pad=0.04, label='Slip')

            # Calculate error metrics if both predicted and ground truth are available
            if gt_slip is not None:
                error_diff = pred_slip - gt_slip
                error_metrics = {
                    "max_error": float(np.max(np.abs(error_diff))),
                    "mean_error": float(np.mean(np.abs(error_diff))),
                    "min_error": float(np.min(np.abs(error_diff))),
                    "std_error": float(np.std(error_diff))
                }
                
                # Save error metrics to JSON file for each image
                if image_name:
                    metrics_dir = "error_metrics"
                    os.makedirs(metrics_dir, exist_ok=True)
                    metrics_path = os.path.join(metrics_dir, f"{image_name}_error_metrics.json")
                    
                    import json
                    with open(metrics_path, 'w') as f:
                        json.dump(error_metrics, f, indent=4)
                
                mean_error = error_metrics["mean_error"]
                fig.suptitle(f"Earthquake: {image_name}")
            else:
                fig.suptitle(f"Earthquake: {image_name}")
        else:
            # Fallback: original visualization without scaling
            fig, axes = plt.subplots(1, 2 if gt_array is not None else 1, figsize=(12, 6))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes[0].imshow(pred, cmap='seismic_r')
            axes[0].set_title("Decoded Image")
            axes[0].axis("off")
            if gt_array is not None and len(axes) > 1:
                axes[1].imshow(gt_array, cmap='seismic_r')
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")
                
                # Calculate error metrics on normalized pixel scale
                error_diff = pred - gt_array
                error_metrics = {
                    "max_error": float(np.max(np.abs(error_diff))),
                    "mean_error": float(np.mean(np.abs(error_diff))),
                    "min_error": float(np.min(np.abs(error_diff))),
                    "std_error": float(np.std(error_diff))
                }
                
                # Save error metrics to JSON file for each image
                if image_name:
                    metrics_dir = "error_metrics"
                    os.makedirs(metrics_dir, exist_ok=True)
                    metrics_path = os.path.join(metrics_dir, f"{image_name}_error_metrics.json")
                    
                    import json
                    with open(metrics_path, 'w') as f:
                        json.dump(error_metrics, f, indent=4)

        # Compute and save metrics on normalized scale
        metrics_json_path = r'test_metrics.json'
        if true_image_path and metrics_json_path:
            save_metrics_for_image(decoded_image,
                                   true_image_path,
                                   metrics_json_path)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
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
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)  # shape: [1, 2704]
        reconstructed_image = decoder(embedding_tensor)  # Use forward() directly
        actual_image_path = rf"Dataset\filtered_images_train\interpolated_slip_image_{key}.fsp.png"
        save_path = rf"Dataset/reconstructed_images/reconstructed_image_{key}.png"
        decoder.visualize_prediction(reconstructed_image, true_image_path=actual_image_path, save_path=save_path)