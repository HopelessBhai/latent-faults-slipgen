import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
from train_pipeline import prepare_dataloaders, train, evaluate  # Ensure these functions are importable
from decoder import Decoder
from input_to_image_embedd import LatentNN

# Global constants (adjust as necessary)
TEXT_EMBED_PATH = r"Dataset/text_vec.npy"
IMAGE_DIR = r"Dataset/filtered_images_train"
OUTPUT_DIM = 2704    # Fixed output dimension
BATCH_SIZE = 32
TEST_SPLIT = 0.2

def objective(trial):
    """
    Objective function that Optuna will minimize. This function sets up your data
    loaders and models, picks hyperparameters using trial.suggest_* methods, trains
    the models for a reduced number of epochs, and returns the final validation loss.
    """
    # Sample hyperparameters from search space.
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True)
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.2)
    lambda_l1 = trial.suggest_float("lambda_l1", 1e-9, 1e-2, log=True)
    # Choose number of hidden layers
    # n_layers = trial.suggest_int("n_layers", 1, 4)
    n_layers = 1  # Fixed number of layers for simplicity in this example
    
    # Generate hidden dimensions for each layer
    hidden_dims = []
    for i in range(n_layers):
        # Decreasing size for each subsequent layer
        min_size = max(16, 128 // (2**i))
        max_size = max(32, 256 // (2**i))
        step = max(8, 32 // (2**i))
        hidden_dims.append(trial.suggest_int(f"hidden_layer_{i+1}", min_size, max_size, step=step))

    # Prepare data
    train_loader, val_loader = prepare_dataloaders(TEXT_EMBED_PATH, IMAGE_DIR, batch_size=BATCH_SIZE, test_split=TEST_SPLIT)

    # Initialize models. The Decoder is assumed to be a pre-trained module.
    decoder = Decoder(model_weights_path=r"models/vqvae_finetuned.pth")
    # Infer input dimension from TEXT_EMBED_PATH
    def _infer_input_dim(npy_path: str) -> int:
        data = np.load(npy_path, allow_pickle=True).item()
        try:
            first_key = next(iter(data))
        except StopIteration:
            raise ValueError(f"No entries found in {npy_path}")
        arr = np.asarray(data[first_key]).reshape(-1)
        return int(arr.shape[0])

    inferred_input_dim = _infer_input_dim(TEXT_EMBED_PATH)
    latent = LatentNN(input_dim=inferred_input_dim, hidden_dims=hidden_dims, output_dim=OUTPUT_DIM, dropout_prob=dropout_prob)

    # Set loss and optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(latent.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Train the model using a smaller number of epochs to make tuning faster.
    tuning_epochs = 500
    early_stopping_patience = 60
    print(f"\nTrial with: lr={learning_rate:.5f}, dropout={dropout_prob:.2f}, lambda_l1={lambda_l1:.7f}, hidden_dims={hidden_dims}")

    # Create unique file prefix for this trial to avoid concurrent access issues
    trial_prefix = f"trial_{trial.number}_"
    
    train(latent, decoder, train_loader, val_loader, criterion, optimizer,
          epochs=tuning_epochs, patience=early_stopping_patience, lambda_l1=lambda_l1,
          save_models=False, model_save_prefix=trial_prefix)  # Disable saving during tuning

    # Evaluate the tuned model on the validation set.
    final_val_loss = evaluate(latent, decoder, val_loader, criterion, lambda_l1)
    print(f"Trial finished with validation loss: {final_val_loss:.4f}\n")
    return final_val_loss

def main():
    # Create an Optuna study to minimize the objective (validation loss).
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)  # You can increase n_trials for a more extensive search.

    # Retrieve and print the best trial.
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {:.4f}".format(trial.value))
    print("  Params:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save the best hyperparameters in JSON format.
    best_params = trial.params
    best_params["val_loss"] = trial.value
    os.makedirs("models", exist_ok=True)
    best_params_path = r"models/best_hyperparams.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best hyperparameters saved to {best_params_path}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds.")