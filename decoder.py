import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from image_embedder import VQVAE  # Assumes PyTorch model is here
from assets.utils import save_metrics_for_image, pixels_to_slip
import pandas as pd
import io
from scipy.interpolate import griddata

# =======================================
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

    def get_lat_lon_from_image(self,interpolated_image, gt_slip,original_x_ew, original_y_ns, 
                            epicenter_lat, epicenter_lon, Dx,src_df):
        """
        Reverses the interpolation process to generate LAT and LON images.

        Args:
            interpolated_image (np.ndarray): The 2D numpy array of the interpolated slip.
            original_x_ew (np.ndarray): The original 'X==EW' coordinates in km.
            original_y_ns (np.ndarray): The original 'Y==NS' coordinates in km.
            epicenter_lat (float): Latitude of the epicenter.
            epicenter_lon (float): Longitude of the epicenter.
            Dx (float): The inversion parameter 'Dx' in km.

        Returns:
            tuple: A tuple containing two 2D numpy arrays: (lat_grid, lon_grid).
        """

                
        # =============================================================================
        # 2. FORWARD PROCESS: Generate the interpolated image (as in your example)
        #    This gives us an image to work with for the reversal.
        # =============================================================================

        # Extract and scale coordinates
        es = src_df['X==EW'].values * Dx / 5.0
        ns = src_df['Y==NS'].values * Dx / 5.0
        slip = src_df['SLIP'].values / src_df['SLIP'].max()

        # Recenter coordinates
        es_centered = es - np.mean(es)
        ns_centered = ns - np.mean(ns)


        # epicenter_lat = input[input['filename']==i]['LAT'].values[0]
        # epicenter_lon = input[input['filename']==i]['LON'].values[0]

        # # Define the grid for interpolation
        print(f"es_centered: {es_centered.shape}")
        x_min = int(np.floor(np.min(es_centered)))
        x_max = int(np.ceil(np.max(es_centered)))
        y_min = int(np.floor(np.min(ns_centered)))
        y_max = int(np.ceil(np.max(ns_centered)))

        grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max + 1),
                                    np.arange(y_min, y_max + 1))

        # Interpolate slip values onto the grid
        points = np.column_stack((es_centered, ns_centered))
        interpolated_slip_image = griddata(points, slip, (grid_x, grid_y), method='cubic', fill_value=0)
        interpolated_slip_image_shape=interpolated_slip_image.shape
        print(interpolated_image.shape,interpolated_slip_image_shape)
        a,b= interpolated_slip_image_shape

        print(type(interpolated_image))

        interpolated_image_pil = Image.fromarray((interpolated_image))
        interpolated_image= interpolated_image_pil.resize((a,b), Image.LANCZOS)
        interpolated_image = np.array(interpolated_image)

        gt_slip_pil = Image.fromarray((gt_slip))
        gt_slip= gt_slip_pil.resize((a,b), Image.LANCZOS)
        gt_slip = np.array(gt_slip)


        # interpolated_slip_image = plt.imread(f"./Dataset/predicted_images_3.0/reconstructed_image_{i[:-4]}.png")
        # if len(interpolated_slip_image.shape) == 3:
        #     interpolated_slip_image = interpolated_slip_image[:, :, 0]
        # print(interpolated_slip_image.shape)    

        # print(f"Generated an example interpolated image with shape: {interpolated_slip_image.shape}")

        # --- Step 1: Recalculate key parameters from the forward process ---
        # These are needed to correctly map pixel coordinates back.
        es = original_x_ew * Dx / 5.0
        ns = original_y_ns * Dx / 5.0
        mean_es = np.mean(es)
        mean_ns = np.mean(ns)
        es_centered = es - mean_es
        ns_centered = ns - mean_ns
        x_min_fwd = int(np.floor(np.min(es_centered)))
        y_min_fwd = int(np.floor(np.min(ns_centered)))
        
        # --- Step 2: Create a grid of the image's pixel coordinates ---
        height, width = interpolated_slip_image_shape
        pixel_cols, pixel_rows = np.meshgrid(np.arange(width), np.arange(height))

        # --- Step 3: Map pixel coordinates back to the centered local grid ---
        # This reverses the creation of the grid.
        es_centered_grid = pixel_cols + x_min_fwd
        ns_centered_grid = pixel_rows + y_min_fwd

        # --- Step 4: Reverse the centering and scaling ---
        # Add the mean back to reverse the centering
        es_grid = es_centered_grid + mean_es
        ns_grid = ns_centered_grid + mean_ns
        
        # Divide by the scaling factor to get back to original km units
        x_ew_grid_km = es_grid / (Dx / 5.0)
        y_ns_grid_km = ns_grid / (Dx / 5.0)
        
        # --- Step 5: Convert local km coordinates to geographic LAT/LON degrees ---
        # The origin (0,0) of the (X==EW, Y==NS) system is the epicenter.
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.radians(epicenter_lat))
        
        # Calculate the offset in degrees from the epicenter
        delta_lat_grid = y_ns_grid_km / km_per_deg_lat
        delta_lon_grid = x_ew_grid_km / km_per_deg_lon
        
        # Add the offset to the epicenter's coordinates
        lat_grid = epicenter_lat + delta_lat_grid
        lon_grid = epicenter_lon + delta_lon_grid
        
        return lat_grid, lon_grid, interpolated_image, gt_slip



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
            input = pd.read_csv(r'./Dataset/extracted_dataset/non-multisegment/non-multisegment_input.csv')
            src = pd.read_csv(r'./Dataset/extracted_dataset/non-multisegment/non-multisegment_output.csv')

            # --- Execute the reversal function ---
            print(src['filename'].unique()[0])
            lat_image, lon_image, extrapolated_pred_slip, extrapolated_gt_slip = self.get_lat_lon_from_image(
                interpolated_image=pred_slip,
                gt_slip=gt_slip,
                original_x_ew=src[src['filename']==(image_name+'.fsp')]['X==EW'].values,
                original_y_ns=src[src['filename']==(image_name+'.fsp')]['Y==NS'].values,
                epicenter_lat=input[input['filename']==(image_name+'.fsp')]['LAT'].values[0],
                epicenter_lon=input[input['filename']==(image_name+'.fsp')]['LON'].values[0],
                Dx=input[input['filename']==(image_name+'.fsp')]['Dx'].values[0],
                src_df=src[src['filename']==(image_name+'.fsp')]
            )
            src_df=src[src['filename']==image_name]
            
            print(f"Generated LAT/LON images, each with shape: {lat_image.shape}")
            # =============================================================================
            # 4. USAGE AND VISUALIZATION
            # =============================================================================
            # --- Visualizing the resulting coordinate images ---
            # fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            FIG_WIDTH = 16
            FIG_HEIGHT = 6
            DPI = 100
            fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)


            # --- Plot 1: Interpolated Slip Image (Prediction) ---
            ax1 = axes[0]
            im1 = ax1.imshow(extrapolated_pred_slip, origin='lower', cmap='viridis', 
                            extent=[lon_image.min(), lon_image.max(), lat_image.min(), lat_image.max()])
            ax1.set_title("Predicted Slip Image")
            ax1.set_xlabel("Longitude")
            ax1.set_ylabel("Latitude")

            # --- Plot 2: Interpolated Slip Image (Ground Truth) ---
            ax2 = axes[1]
            im2 = ax2.imshow(extrapolated_gt_slip, origin='lower', cmap='viridis', 
                            extent=[lon_image.min(), lon_image.max(), lat_image.min(), lat_image.max()])
            ax2.set_title("Ground Truth Slip Image")
            ax2.set_xlabel("Longitude")
            ax2.set_ylabel("Latitude")

            # --- Add a single colorbar for the entire figure ---4
            fig.colorbar(im2, ax=axes[0], label='Slip', pad=0.04, aspect=30)
            fig.colorbar(im2, ax=axes[1], label='Slip', pad=0.04, aspect=30)


            # plt.show()

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
                fig.suptitle(f"Earthquake: {input[input['filename']==(image_name+'.fsp')]['Event'].values[0]}")
            else:
                fig.suptitle(f"Earthquake: {input[input['filename']==(image_name+'.fsp')]['Event'].values[0]}")
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
            print("No save path found....")
            plt.show()
        plt.close()
        return extrapolated_pred_slip

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