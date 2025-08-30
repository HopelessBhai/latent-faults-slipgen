#!/usr/bin/env python3
"""
Script to analyze text_vec.npy and understand the parameter structure
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the text embeddings
print("Loading text_vec.npy...")
text_embed = np.load(r'Dataset/text_vec.npy', allow_pickle=True).item()

print(f"Number of samples: {len(text_embed)}")
print(f"Keys (first 5): {list(text_embed.keys())[:5]}")

# Get the first sample to understand structure
first_key = list(text_embed.keys())[0]
first_sample = text_embed[first_key]

print(f"\nFirst sample key: {first_key}")
print(f"First sample shape: {first_sample.shape}")
print(f"First sample values: {first_sample}")

# Collect all embeddings into a matrix
all_embeddings = []
all_keys = []
for key, embedding in text_embed.items():
    all_embeddings.append(embedding)
    all_keys.append(key)

embeddings_matrix = np.array(all_embeddings)
print(f"\nEmbeddings matrix shape: {embeddings_matrix.shape}")

# Calculate statistics for each parameter
param_stats = {}
for i in range(embeddings_matrix.shape[1]):
    param_values = embeddings_matrix[:, i]
    param_stats[f'param_{i}'] = {
        'min': float(np.min(param_values)),
        'max': float(np.max(param_values)),
        'mean': float(np.mean(param_values)),
        'std': float(np.std(param_values)),
        'values': param_values
    }

print(f"\nParameter statistics:")
for param_name, stats in param_stats.items():
    print(f"{param_name}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")

# Save analysis results
print(f"\nSaving analysis results...")
analysis_results = {
    'num_samples': len(text_embed),
    'num_parameters': embeddings_matrix.shape[1],
    'sample_keys': all_keys[:10],  # First 10 keys as examples
    'parameter_stats': {k: {key: val for key, val in v.items() if key != 'values'} 
                       for k, v in param_stats.items()},
    'embeddings_matrix': embeddings_matrix
}

np.save('text_vec_analysis.npy', analysis_results, allow_pickle=True)
print("Analysis saved to text_vec_analysis.npy")

# Create visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(min(7, embeddings_matrix.shape[1])):
    ax = axes[i]
    ax.hist(embeddings_matrix[:, i], bins=30, alpha=0.7, edgecolor='black')
    ax.set_title(f'Parameter {i}\nRange: [{param_stats[f"param_{i}"]["min"]:.2f}, {param_stats[f"param_{i}"]["max"]:.2f}]')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# Remove unused subplot
if embeddings_matrix.shape[1] < 8:
    for i in range(embeddings_matrix.shape[1], 8):
        fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('parameter_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("Parameter distribution plot saved as parameter_distributions.png")
