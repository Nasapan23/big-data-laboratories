"""
Clustering Module
Step 6: KMeans clustering and cluster analysis
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def apply_kmeans(lsi_vectors, n_clusters=7, random_state=42):
    """
    Apply KMeans clustering on LSI vectors.
    
    Args:
        lsi_vectors: LSI vectors (n_samples, n_features)
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (cluster_labels, kmeans_model)
            - cluster_labels: Cluster assignments for each sample
            - kmeans_model: Fitted KMeans model
    """
    print(f"Applying KMeans clustering with {n_clusters} clusters...")
    
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans_model.fit_predict(lsi_vectors)
    
    print(f"Clustering complete. Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} incidents")
    
    return cluster_labels, kmeans_model


def analyze_clusters(df, cluster_labels, tfidf_matrix, tfidf_vectorizer, 
                    top_terms=10, examples_per_cluster=5):
    """
    Analyze clusters by showing representative examples and top terms.
    
    Args:
        df: DataFrame with incident data
        cluster_labels: Cluster assignments
        tfidf_matrix: Original TF-IDF matrix
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        top_terms: Number of top terms to display per cluster
        examples_per_cluster: Number of example incidents to show per cluster
        
    Returns:
        Dictionary with cluster analysis results
    """
    df = df.copy()
    df['cluster'] = cluster_labels
    
    n_clusters = len(np.unique(cluster_labels))
    
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS")
    print("="*80)
    
    cluster_analysis = {}
    
    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        
        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_id} ({len(cluster_data)} incidents)")
        print(f"{'='*80}")
        
        # Get representative examples
        print(f"\nRepresentative Examples (showing {min(examples_per_cluster, len(cluster_data))}):")
        examples = cluster_data.head(examples_per_cluster)
        for idx, row in examples.iterrows():
            print(f"  - {row['combined_text']}")
        
        # Get top TF-IDF terms for this cluster
        top_terms_list = get_cluster_top_terms(
            tfidf_matrix,
            cluster_labels,
            tfidf_vectorizer,
            cluster_id,
            top_terms=top_terms
        )
        
        print(f"\nTop {top_terms} TF-IDF Terms:")
        for term, score in top_terms_list:
            print(f"  {term}: {score:.4f}")
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_data),
            'examples': examples[['combined_text', 'Complaint Type', 'Descriptor']].to_dict('records'),
            'top_terms': top_terms_list
        }
    
    return cluster_analysis


def get_cluster_top_terms(tfidf_matrix, cluster_labels, tfidf_vectorizer, 
                          cluster_id, top_terms=10):
    """
    Get top TF-IDF terms for a specific cluster.
    
    Args:
        tfidf_matrix: Original TF-IDF matrix
        cluster_labels: Cluster assignments
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        cluster_id: Cluster ID to analyze
        top_terms: Number of top terms to return
        
    Returns:
        List of (term, score) tuples
    """
    # Get indices of incidents in this cluster
    cluster_mask = cluster_labels == cluster_id
    cluster_tfidf = tfidf_matrix[cluster_mask]
    
    # Compute mean TF-IDF scores for this cluster
    mean_scores = np.asarray(cluster_tfidf.mean(axis=0)).flatten()
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Get top terms
    top_indices = np.argsort(mean_scores)[::-1][:top_terms]
    top_terms_list = [(feature_names[idx], mean_scores[idx]) for idx in top_indices]
    
    return top_terms_list
