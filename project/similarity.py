"""
Semantic Similarity Search Module
Step 5: Find semantically similar incidents using cosine similarity
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from preprocessing import preprocess_text


def find_similar_incidents(query_text, tfidf_vectorizer, lsi_model, lsi_vectors, 
                          df, top_k=5):
    """
    Find top-k most semantically similar incidents to a query.
    
    Args:
        query_text: Query text string (e.g., "Noise - Residential")
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        lsi_model: Fitted LSI (TruncatedSVD) model
        lsi_vectors: LSI vectors for all incidents (for comparison)
        df: DataFrame with incident data
        top_k: Number of similar incidents to return
        
    Returns:
        List of dictionaries, each containing:
            - index: Original DataFrame index
            - similarity: Cosine similarity score
            - combined_text: Original combined text
            - complaint_type: Complaint type
            - descriptor: Descriptor
    """
    # Preprocess query text
    preprocessed_query = preprocess_text(query_text)
    
    # Transform query using TF-IDF
    query_tfidf = tfidf_vectorizer.transform([preprocessed_query])
    
    # Transform query using LSI
    query_lsi = lsi_model.transform(query_tfidf)
    
    # Compute cosine similarity with all incidents
    similarities = cosine_similarity(query_lsi, lsi_vectors)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build results
    results = []
    for idx in top_indices:
        result = {
            'index': int(idx),
            'similarity': float(similarities[idx]),
            'combined_text': df.iloc[idx]['combined_text'],
            'complaint_type': df.iloc[idx]['Complaint Type'],
            'descriptor': df.iloc[idx]['Descriptor']
        }
        
        # Add Unique Key if available
        if 'Unique Key' in df.columns:
            result['unique_key'] = str(df.iloc[idx]['Unique Key'])
        
        # Add Created Date if available
        if 'Created Date' in df.columns:
            result['created_date'] = str(df.iloc[idx]['Created Date'])
        
        # Add location fields if available
        if 'Incident Address' in df.columns:
            result['incident_address'] = str(df.iloc[idx]['Incident Address']) if pd.notna(df.iloc[idx]['Incident Address']) else ''
        if 'Street Name' in df.columns:
            result['street_name'] = str(df.iloc[idx]['Street Name']) if pd.notna(df.iloc[idx]['Street Name']) else ''
        if 'Incident Zip' in df.columns:
            result['incident_zip'] = str(df.iloc[idx]['Incident Zip']) if pd.notna(df.iloc[idx]['Incident Zip']) else ''
        if 'Borough' in df.columns:
            result['borough'] = str(df.iloc[idx]['Borough']) if pd.notna(df.iloc[idx]['Borough']) else ''
        if 'City' in df.columns:
            result['city'] = str(df.iloc[idx]['City']) if pd.notna(df.iloc[idx]['City']) else ''
        
        results.append(result)
    
    return results


def analyze_locations(results, top_n=10):
    """
    Analyze location patterns in search results.
    
    Args:
        results: List of result dictionaries from find_similar_incidents()
        top_n: Number of top locations to return
        
    Returns:
        Dictionary with location statistics:
            - borough_counts: Count by borough
            - street_counts: Count by street (top N)
            - zip_counts: Count by zip code (top N)
            - total: Total number of results
    """
    if not results:
        return {
            'total': 0,
            'borough_counts': {},
            'street_counts': {},
            'zip_counts': {}
        }
    
    # Count by borough
    borough_counts = {}
    street_counts = {}
    zip_counts = {}
    
    for result in results:
        # Count boroughs
        if 'borough' in result and result['borough']:
            borough = result['borough']
            borough_counts[borough] = borough_counts.get(borough, 0) + 1
        
        # Count streets
        if 'street_name' in result and result['street_name']:
            street = result['street_name']
            street_counts[street] = street_counts.get(street, 0) + 1
        
        # Count zip codes
        if 'incident_zip' in result and result['incident_zip']:
            zip_code = result['incident_zip']
            zip_counts[zip_code] = zip_counts.get(zip_code, 0) + 1
    
    # Sort and get top N
    sorted_streets = sorted(street_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    sorted_zips = sorted(zip_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return {
        'total': len(results),
        'borough_counts': borough_counts,
        'street_counts': dict(sorted_streets),
        'zip_counts': dict(sorted_zips)
    }
