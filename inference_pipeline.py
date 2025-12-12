import torch
from input_to_image_embedd import LatentNN
from decoder import Decoder
import json
import numpy as np
import pickle
import os
from PIL import Image
from assets.utils import pixels_to_slip

class Inference:
    def __init__(self, 
                 latent_model_path=r"models/latent_model.pth",
                 decoder_model_path=r"models/decoder_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate both components with the appropriate hyperparameters.
        with open(r'models/best_hyperparams.json', 'r') as f:
            hyperparams = json.load(f)
        dropout_prob = hyperparams["dropout_prob"]
        hidden_dims = [hyperparams[f"hidden_layer_{i}"] for i in range(1, 1+1)]

        # Infer input dimension from Dataset/text_vec.npy
        def _infer_input_dim(npy_path: str) -> int:
            data = np.load(npy_path, allow_pickle=True).item()
            try:
                first_key = next(iter(data))
            except StopIteration:
                raise ValueError(f"No entries found in {npy_path}")
            arr = np.asarray(data[first_key]).reshape(-1)
            return int(arr.shape[0])

        input_dim = _infer_input_dim(r"Dataset/text_vec.npy")
        output_dim = 2704 # fixed output dim

        # Create latent model and load its weights.
        self.latent_model = LatentNN(input_dim=input_dim, hidden_dims=hidden_dims, 
                                     output_dim=output_dim, dropout_prob=dropout_prob)
        self.latent_model.load_state_dict(torch.load(latent_model_path, map_location=self.device))
        print("Loaded latent model weights from:", latent_model_path)

        # Create the decoder and load its weights.
        self.decoder = Decoder(model_weights_path=decoder_model_path)
        print("Loaded decoder weights from:", decoder_model_path)
        
        self.latent_model.to(self.device).eval()
        self.decoder.to(self.device).eval()
        print("Models are set to evaluation mode.")

        # Load Dz map once
        dz_path = os.path.join('assets', 'dz.json')
        if os.path.isfile(dz_path):
            with open(dz_path, 'r') as f:
                self.dz_by_key = json.load(f)
        else:
            self.dz_by_key = {}

    def generate(self, text, actual_image_path=None, save_path=None, show_plot=True):
        """
        Generate an image from the provided text using the end-to-end trained model.
        """
        # Step 1: Text to embedding. (For demonstration, we load from file.)
        text_embed_dict = np.load(r'Dataset/text_vec.npy', allow_pickle=True).item()
        # Here, 'text' is assumed to be a key in the dictionary.
        text_embedding = text_embed_dict[text]

        # Load the scaler and transform the text embedding.
        with open('scaler_x.pkl', 'rb') as f:
            loaded_scaler_x = pickle.load(f)
        # Transform the text embedding using the loaded scaler.
        text_embedding = loaded_scaler_x.transform(text_embedding.reshape(1, -1))
        
        # Step 2: Map to image latent using the latent model.
        with torch.no_grad():
            input_tensor = torch.tensor(text_embedding, dtype=torch.float32).to(self.device)
            if input_tensor.ndim == 1:
                input_tensor = input_tensor.unsqueeze(0)
            # Get the latent representation.
            image_latent = self.latent_model(input_tensor)
            
            # Step 3: Decode the latent representation. 
            predicted_image_tensor = self.decoder(image_latent)

        # Step 4: Convert to slip values and save as numpy array
        # Extract key name (without .fsp) to match dz.json keys
        image_key = text
        if image_key.endswith('.fsp'):
            image_key = image_key[:-4]
        dz = self.dz_by_key.get(image_key)

        # Convert predicted image to slip values and save as numpy array
        if dz is not None:
            # Get the predicted image as numpy array
            pred_array = predicted_image_tensor[0, 0].detach().cpu().numpy()
            # Normalize if necessary
            if pred_array.max() > 1.0 or pred_array.min() < 0.0:
                pred_array = np.clip(pred_array, 0.0, 1.0)

            # Convert to slip values
            slip_array = pixels_to_slip(pred_array, dz, image_name=image_key, plot=False)

        # if show_plot:
        extrapolated_pred_slip=self.decoder.visualize_prediction(
            predicted_image_tensor,
            true_image_path=actual_image_path,
            save_path=save_path,
            dz=dz,
            image_name=image_key
        )
        # Save slip array as numpy file
        slip_array = np.array(extrapolated_pred_slip)
        slip_save_dir = "Dataset/slip_arrays_inference"
        os.makedirs(slip_save_dir, exist_ok=True)
        slip_save_path = os.path.join(slip_save_dir, f"slip_array_{image_key}.npy")
        np.save(slip_save_path, slip_array)
        print(f"Saved slip array to: {slip_save_path}")

        return predicted_image_tensor

# Example usage:
if __name__ == "__main__":
    generator = Inference()

    input_file = np.load(r'Dataset/text_vec.npy', allow_pickle=True).item()

    # Folder containing the actual images
    actual_images_folder = r"Dataset/filtered_images_test"


    # Get all image files in the folder
    for image_file in os.listdir(actual_images_folder):
        if image_file.startswith("interpolated_slip_image_") and image_file.endswith(".png"):
            # Extract the key from the image filename
            key = image_file.replace("interpolated_slip_image_", "").replace(".fsp.png", "")  # Remove extension
            
            if key in input_file:
                save_path = rf"Dataset/predicted_images_LAT_LON/reconstructed_image_{key}.png"
                text_input = input_file[key]
                
                # Generate the image using the key from actual image path
                output_image = generator.generate(text=key, 
                                                  actual_image_path=os.path.join(actual_images_folder, image_file),
                                                  save_path=save_path, show_plot=False)
                print(f"Generated image for key: {key}")
                
            else:
                print(f"No text found for key: {key}")
                