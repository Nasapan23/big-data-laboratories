"""
Latent Semantic Indexing Module
Step 4: Apply TruncatedSVD for LSI dimensionality reduction
"""

from sklearn.decomposition import TruncatedSVD
import numpy as np


def apply_lsi(tfidf_matrix, n_components=150, random_state=42):
    """
    Apply Latent Semantic Indexing (LSI) using TruncatedSVD.
    
    Args:
        tfidf_matrix: TF-IDF matrix (sparse or dense)
        n_components: Number of components (latent dimensions) to keep
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (lsi_vectors, lsi_model)
            - lsi_vectors: Reduced-dimensionality vectors
            - lsi_model: Fitted TruncatedSVD model
    """
    # Get the actual number of features
    n_features = tfidf_matrix.shape[1]
    
    # Adjust n_components if it's greater than the number of features
    actual_n_components = min(n_components, n_features - 1)  # -1 to be safe
    
    if actual_n_components < n_components:
        print(f"Warning: Requested {n_components} components but only {n_features} features available.")
        print(f"Using {actual_n_components} components instead.")
    else:
        print(f"Applying LSI with {actual_n_components} components...")
    
    lsi_model = TruncatedSVD(
        n_components=actual_n_components,
        random_state=random_state,
        algorithm='randomized',
        n_iter=5
    )
    
    lsi_vectors = lsi_model.fit_transform(tfidf_matrix)
    
    print(f"LSI vectors shape: {lsi_vectors.shape}")
    print(f"Explained variance ratio: {lsi_model.explained_variance_ratio_.sum():.4f}")
    
    return lsi_vectors, lsi_model
