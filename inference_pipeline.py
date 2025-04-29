import torch
from input_to_image_embedd import LatentNN
from decoder import Decoder
import json
import numpy as np
import pickle
import os

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
        input_dim = 35   # fixed input dim
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

        # Step 4: Visualize the prediction using the decoder's visualization utility.
        self.decoder.visualize_prediction(predicted_image_tensor, 
                                          true_image_path=actual_image_path, save_path=save_path)
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
                save_path = rf"Dataset/predicted_images/reconstructed_image_{key}.png"
                text_input = input_file[key]
                
                # Generate the image using the key from actual image path
                output_image = generator.generate(text=key, 
                                                  actual_image_path=os.path.join(actual_images_folder, image_file),
                                                  save_path=save_path)
                print(f"Generated image for key: {key}")
                
            else:
                print(f"No text found for key: {key}")