"""
Main Script
Orchestrates the entire NLP pipeline for semantic analysis of NYC 311 service requests
"""

import os
import pickle
from data_loading import load_data
from preprocessing import preprocess_dataframe
from vectorization import vectorize_texts
from lsi import apply_lsi
from similarity import find_similar_incidents, analyze_locations
from clustering import apply_kmeans, analyze_clusters, get_cluster_top_terms
from visualization import reduce_to_2d, plot_clusters
from reporting import generate_html_report
from persistence import (
    create_output_directories,
    save_tfidf_vectors,
    save_lsi_vectors,
    save_processed_data,
    save_cluster_assignments,
    save_models,
    save_metadata
)


def main():
    """Main pipeline execution"""
    
    # Configuration
    CSV_PATH = 'dataset.csv'
    N_COMPONENTS = 150  # LSI components
    N_CLUSTERS = 7  # Number of clusters
    MAX_FEATURES = 5000  # TF-IDF max features
    MIN_DF = 2  # TF-IDF minimum document frequency
    RANDOM_STATE = 42
    
    print("="*80)
    print("NLP PIPELINE: Semantic Analysis of NYC 311 Service Requests")
    print("="*80)
    
    # Create output directories
    create_output_directories()
    
    # Step 1: Data Loading
    print("\n[Step 1] Loading data...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file '{CSV_PATH}' not found!")
        return
    
    df = load_data(CSV_PATH)
    print(f"Loaded {len(df)} records")
    
    # Step 2: Text Preprocessing
    print("\n[Step 2] Preprocessing text...")
    df = preprocess_dataframe(df)
    print("Text preprocessing complete")
    
    # Save processed data
    save_processed_data(df)
    
    # Step 3: TF-IDF Vectorization
    print("\n[Step 3] Creating TF-IDF vectors...")
    tfidf_matrix, tfidf_vectorizer = vectorize_texts(
        df['preprocessed_text'],
        max_features=MAX_FEATURES,
        min_df=MIN_DF
    )
    
    # Save TF-IDF vectors
    save_tfidf_vectors(tfidf_matrix)
    
    # Step 4: LSI
    print("\n[Step 4] Applying Latent Semantic Indexing...")
    lsi_vectors, lsi_model = apply_lsi(
        tfidf_matrix,
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE
    )
    
    # Save LSI vectors
    save_lsi_vectors(lsi_vectors)
    
    # Step 5: Clustering
    print("\n[Step 5] Applying KMeans clustering...")
    cluster_labels, kmeans_model = apply_kmeans(
        lsi_vectors,
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE
    )
    
    # Save cluster assignments
    save_cluster_assignments(df, cluster_labels)
    
    # Step 6: Cluster Analysis
    print("\n[Step 6] Analyzing clusters...")
    cluster_analysis = analyze_clusters(
        df,
        cluster_labels,
        tfidf_matrix,
        tfidf_vectorizer,
        top_terms=10,
        examples_per_cluster=5
    )
    
    # Step 7: Visualization
    print("\n[Step 7] Creating 2D visualization...")
    # Use t-SNE by default as it's faster for large datasets
    # Set method='umap' if you prefer UMAP (slower but sometimes better quality)
    coordinates_2d, sample_indices = reduce_to_2d(
        lsi_vectors,
        method='tsne',  # Changed to t-SNE for faster computation
        random_state=RANDOM_STATE,
        max_samples=30000  # Reduced from 50000 for faster processing
    )
    viz_path = os.path.join('output', 'visualizations', 'cluster_visualization.png')
    plot_clusters(coordinates_2d, cluster_labels, sample_indices=sample_indices, save_path=viz_path)
    
    # Step 8: Demonstrate Similarity Search
    print("\n" + "="*80)
    print("SEMANTIC SIMILARITY SEARCH EXAMPLES")
    print("="*80)
    
    example_queries = [
        "Noise - Residential",
        "Illegal Parking - Blocked Driveway",
        "Noise - Street/Sidewalk"
    ]
    
    similarity_results = {}
    for query in example_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        similar = find_similar_incidents(
            query,
            tfidf_vectorizer,
            lsi_model,
            lsi_vectors,
            df,
            top_k=5
        )
        similarity_results[query] = similar
        for i, result in enumerate(similar, 1):
            print(f"{i}. Similarity: {result['similarity']:.4f}")
            print(f"   {result['combined_text']}")
            if 'unique_key' in result:
                print(f"   ID: {result['unique_key']}")
            if 'created_date' in result:
                print(f"   Date: {result['created_date']}")
            # Location information
            location_parts = []
            if 'incident_address' in result and result.get('incident_address'):
                location_parts.append(f"Address: {result['incident_address']}")
            if 'street_name' in result and result.get('street_name'):
                location_parts.append(f"Street: {result['street_name']}")
            if 'borough' in result and result.get('borough'):
                location_parts.append(f"Borough: {result['borough']}")
            if 'incident_zip' in result and result.get('incident_zip'):
                location_parts.append(f"Zip: {result['incident_zip']}")
            if location_parts:
                print(f"   Location: {' | '.join(location_parts)}")
        
        # Location analysis
        if similar:
            location_stats = analyze_locations(similar, top_n=5)
            if location_stats['total'] > 0:
                print("\n   Location Analysis:")
                if location_stats['borough_counts']:
                    sorted_boroughs = sorted(location_stats['borough_counts'].items(), 
                                           key=lambda x: x[1], reverse=True)
                    borough_str = ", ".join([f"{b}({c})" for b, c in sorted_boroughs[:5]])
                    print(f"     Boroughs: {borough_str}")
                if location_stats['street_counts']:
                    street_str = ", ".join([f"{s}({c})" for s, c in list(location_stats['street_counts'].items())[:3]])
                    print(f"     Top Streets: {street_str}")
    
    # Step 9: Generate HTML Report
    print("\n[Step 9] Generating HTML report...")
    report_path = os.path.join('output', 'reports', 'analysis_report.html')
    generate_html_report(
        cluster_analysis,
        df,
        cluster_labels,
        cluster_viz_path=os.path.join('output', 'visualizations', 'cluster_visualization.png'),
        similarity_results=similarity_results,
        output_path=report_path
    )
    
    # Step 10: Save models
    print("\n[Step 10] Saving models...")
    save_models(tfidf_vectorizer, lsi_model, kmeans_model)
    
    # Also save legacy models.pkl for backward compatibility with search tools
    legacy_models = {
        'tfidf_vectorizer': tfidf_vectorizer,
        'lsi_model': lsi_model,
        'lsi_vectors': lsi_vectors,
        'df': df
    }
    legacy_path = os.path.join('output', 'models', 'models.pkl')
    with open(legacy_path, 'wb') as f:
        pickle.dump(legacy_models, f)
    print(f"Saved legacy models.pkl to: {legacy_path}")
    
    # Step 11: Save metadata
    print("\n[Step 11] Saving metadata...")
    config = {
        'max_features': MAX_FEATURES,
        'min_df': MIN_DF,
        'lsi_components': N_COMPONENTS,
        'n_clusters': N_CLUSTERS,
        'random_state': RANDOM_STATE
    }
    dataset_info = {
        'total_records': len(df),
        'n_features_tfidf': tfidf_matrix.shape[1],
        'n_components_lsi': lsi_vectors.shape[1],
        'n_clusters': N_CLUSTERS,
        'vocabulary_size': len(tfidf_vectorizer.vocabulary_)
    }
    save_metadata(config, dataset_info)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Processed {len(df)} incidents")
    print(f"Created {N_CLUSTERS} clusters")
    print(f"\nAll outputs saved to: output/")
    print(f"  - Embeddings: output/embeddings/")
    print(f"  - Data: output/data/")
    print(f"  - Models: output/models/")
    print(f"  - Visualizations: output/visualizations/")
    print(f"  - Reports: output/reports/")


if __name__ == "__main__":
    main()
