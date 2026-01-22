"""
Visualization Module
Step 7: 2D visualization using UMAP or t-SNE
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress UMAP warnings about spectral initialization
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available, will use t-SNE instead")

from sklearn.manifold import TSNE


def reduce_to_2d(lsi_vectors, method='umap', random_state=42, n_components=2, max_samples=50000):
    """
    Reduce LSI vectors to 2D for visualization.
    
    Args:
        lsi_vectors: LSI vectors (n_samples, n_features)
        method: 'umap' or 'tsne'
        random_state: Random seed for reproducibility
        n_components: Number of dimensions (should be 2)
        max_samples: Maximum number of samples to use (for large datasets)
        
    Returns:
        tuple: (coordinates_2d, sample_indices)
            - coordinates_2d: 2D coordinates
            - sample_indices: Indices of samples used (for matching with cluster_labels)
    """
    n_samples = lsi_vectors.shape[0]
    
    # Sample data if it's too large for visualization
    if n_samples > max_samples:
        print(f"Dataset is large ({n_samples:,} samples). Sampling {max_samples:,} samples for visualization...")
        np.random.seed(random_state)
        sample_indices = np.random.choice(n_samples, size=max_samples, replace=False)
        sample_indices = np.sort(sample_indices)  # Keep sorted for easier indexing
        lsi_vectors_sample = lsi_vectors[sample_indices]
    else:
        sample_indices = np.arange(n_samples)
        lsi_vectors_sample = lsi_vectors
    
    if method == 'umap' and UMAP_AVAILABLE:
        print("Using UMAP for dimensionality reduction...")
        # Use simpler settings to avoid spectral initialization issues
        # init='random' forces random initialization, avoiding spectral method
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=10,  # Reduced from 15 for faster computation
                min_dist=0.1,
                metric='euclidean',
                init='random',  # Skip spectral initialization to avoid failures
                n_jobs=1,
                verbose=False  # Suppress additional output
            )
            coordinates_2d = reducer.fit_transform(lsi_vectors_sample)
    else:
        print("Using t-SNE for dimensionality reduction...")
        # Use standard parameters - let sklearn handle iterations
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=min(30, len(lsi_vectors_sample) - 1)  # Perplexity must be < n_samples
        )
        coordinates_2d = reducer.fit_transform(lsi_vectors_sample)
    
    print(f"Reduced to 2D. Shape: {coordinates_2d.shape}")
    return coordinates_2d, sample_indices


def plot_clusters(coordinates_2d, cluster_labels, sample_indices=None, save_path='cluster_visualization.png'):
    """
    Create scatter plot of clusters in 2D space.
    
    Args:
        coordinates_2d: 2D coordinates (n_samples, 2)
        cluster_labels: Cluster assignments (full array or sampled)
        sample_indices: If provided, cluster_labels[sample_indices] will be used
        save_path: Path to save the visualization
    """
    # If sample_indices is provided, use sampled cluster labels
    if sample_indices is not None:
        cluster_labels_plot = cluster_labels[sample_indices]
    else:
        cluster_labels_plot = cluster_labels
    
    n_clusters = len(np.unique(cluster_labels_plot))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for cluster_id in range(n_clusters):
        mask = cluster_labels_plot == cluster_id
        plt.scatter(
            coordinates_2d[mask, 0],
            coordinates_2d[mask, 1],
            c=[colors[cluster_id]],
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            s=20
        )
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    n_shown = len(coordinates_2d)
    title = f'Cluster Visualization (2D Projection of LSI Vectors)'
    if sample_indices is not None:
        title += f'\n(Showing {n_shown:,} sampled points)'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    plt.close()
