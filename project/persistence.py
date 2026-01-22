"""
Data Persistence Module
Functions to save and load embeddings, data, models, and metadata
"""

import os
import json
import pickle
import numpy as np
from scipy import sparse
from datetime import datetime
import pandas as pd

# Base output directory
OUTPUT_DIR = 'output'
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, 'embeddings')
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')


def create_output_directories():
    """Create output directory structure if it doesn't exist"""
    directories = [
        OUTPUT_DIR,
        EMBEDDINGS_DIR,
        DATA_DIR,
        MODELS_DIR,
        VISUALIZATIONS_DIR,
        REPORTS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Created output directory structure: {OUTPUT_DIR}/")


def save_tfidf_vectors(tfidf_matrix, filepath=None):
    """Save TF-IDF matrix as sparse .npz file"""
    if filepath is None:
        filepath = os.path.join(EMBEDDINGS_DIR, 'tfidf_vectors.npz')
    
    # Convert to sparse matrix if needed and save
    if sparse.issparse(tfidf_matrix):
        sparse.save_npz(filepath, tfidf_matrix)
    else:
        # Convert to sparse if it's dense
        sparse_matrix = sparse.csr_matrix(tfidf_matrix)
        sparse.save_npz(filepath, sparse_matrix)
    
    print(f"Saved TF-IDF vectors to: {filepath}")
    return filepath


def load_tfidf_vectors(filepath=None):
    """Load TF-IDF matrix from .npz file"""
    if filepath is None:
        filepath = os.path.join(EMBEDDINGS_DIR, 'tfidf_vectors.npz')
    
    tfidf_matrix = sparse.load_npz(filepath)
    print(f"Loaded TF-IDF vectors from: {filepath}")
    return tfidf_matrix


def save_lsi_vectors(lsi_vectors, filepath=None):
    """Save LSI vectors as .npy file"""
    if filepath is None:
        filepath = os.path.join(EMBEDDINGS_DIR, 'lsi_vectors.npy')
    
    np.save(filepath, lsi_vectors)
    print(f"Saved LSI vectors to: {filepath} (shape: {lsi_vectors.shape})")
    return filepath


def load_lsi_vectors(filepath=None):
    """Load LSI vectors from .npy file"""
    if filepath is None:
        filepath = os.path.join(EMBEDDINGS_DIR, 'lsi_vectors.npy')
    
    lsi_vectors = np.load(filepath)
    print(f"Loaded LSI vectors from: {filepath} (shape: {lsi_vectors.shape})")
    return lsi_vectors


def save_embeddings(tfidf_matrix, lsi_vectors):
    """Save both TF-IDF and LSI vectors"""
    create_output_directories()
    save_tfidf_vectors(tfidf_matrix)
    save_lsi_vectors(lsi_vectors)


def load_embeddings():
    """Load both TF-IDF and LSI vectors"""
    tfidf_matrix = load_tfidf_vectors()
    lsi_vectors = load_lsi_vectors()
    return tfidf_matrix, lsi_vectors


def save_processed_data(df, filepath=None):
    """Save processed DataFrame to CSV"""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'processed_data.csv')
    
    create_output_directories()
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Saved processed data to: {filepath} ({len(df):,} records)")
    return filepath


def load_processed_data(filepath=None):
    """Load processed DataFrame from CSV"""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'processed_data.csv')
    
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Loaded processed data from: {filepath} ({len(df):,} records)")
    return df


def save_cluster_assignments(df, cluster_labels, filepath=None):
    """Save DataFrame with cluster assignments"""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'cluster_assignments.csv')
    
    create_output_directories()
    df_output = df.copy()
    df_output['cluster'] = cluster_labels
    df_output.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Saved cluster assignments to: {filepath}")
    return filepath


def load_cluster_assignments(filepath=None):
    """Load DataFrame with cluster assignments"""
    if filepath is None:
        filepath = os.path.join(DATA_DIR, 'cluster_assignments.csv')
    
    df = pd.read_csv(filepath, encoding='utf-8')
    cluster_labels = df['cluster'].values
    df = df.drop(columns=['cluster'])
    print(f"Loaded cluster assignments from: {filepath}")
    return df, cluster_labels


def save_models(tfidf_vectorizer, lsi_model, kmeans_model=None):
    """Save individual model files"""
    create_output_directories()
    
    # Save TF-IDF vectorizer
    tfidf_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"Saved TF-IDF vectorizer to: {tfidf_path}")
    
    # Save LSI model
    lsi_path = os.path.join(MODELS_DIR, 'lsi_model.pkl')
    with open(lsi_path, 'wb') as f:
        pickle.dump(lsi_model, f)
    print(f"Saved LSI model to: {lsi_path}")
    
    # Save KMeans model if provided
    if kmeans_model is not None:
        kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
        with open(kmeans_path, 'wb') as f:
            pickle.dump(kmeans_model, f)
        print(f"Saved KMeans model to: {kmeans_path}")
    
    return {
        'tfidf_vectorizer': tfidf_path,
        'lsi_model': lsi_path,
        'kmeans_model': os.path.join(MODELS_DIR, 'kmeans_model.pkl') if kmeans_model is not None else None
    }


def load_models():
    """Load all models"""
    # Load TF-IDF vectorizer
    tfidf_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    with open(tfidf_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    print(f"Loaded TF-IDF vectorizer from: {tfidf_path}")
    
    # Load LSI model
    lsi_path = os.path.join(MODELS_DIR, 'lsi_model.pkl')
    with open(lsi_path, 'rb') as f:
        lsi_model = pickle.load(f)
    print(f"Loaded LSI model from: {lsi_path}")
    
    # Try to load KMeans model (optional)
    kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
    kmeans_model = None
    if os.path.exists(kmeans_path):
        with open(kmeans_path, 'rb') as f:
            kmeans_model = pickle.load(f)
        print(f"Loaded KMeans model from: {kmeans_path}")
    
    return tfidf_vectorizer, lsi_model, kmeans_model


def save_metadata(config, dataset_info, filepath=None):
    """Save metadata as JSON"""
    if filepath is None:
        filepath = os.path.join(EMBEDDINGS_DIR, 'metadata.json')
    
    create_output_directories()
    
    metadata = {
        'pipeline_version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'dataset_info': dataset_info,
        'configuration': config,
        'file_paths': {
            'tfidf_vectors': 'embeddings/tfidf_vectors.npz',
            'lsi_vectors': 'embeddings/lsi_vectors.npy',
            'processed_data': 'data/processed_data.csv',
            'cluster_assignments': 'data/cluster_assignments.csv',
            'tfidf_vectorizer': 'models/tfidf_vectorizer.pkl',
            'lsi_model': 'models/lsi_model.pkl',
            'kmeans_model': 'models/kmeans_model.pkl'
        }
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved metadata to: {filepath}")
    return filepath


def load_metadata(filepath=None):
    """Load metadata from JSON"""
    if filepath is None:
        filepath = os.path.join(EMBEDDINGS_DIR, 'metadata.json')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Loaded metadata from: {filepath}")
    return metadata


def load_all_for_search():
    """Convenience function to load everything needed for semantic search"""
    tfidf_vectorizer, lsi_model, _ = load_models()
    lsi_vectors = load_lsi_vectors()
    df = load_processed_data()
    
    return tfidf_vectorizer, lsi_model, lsi_vectors, df
