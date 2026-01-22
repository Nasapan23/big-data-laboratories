"""
Interactive Semantic Search Tool
Allows users to search for similar incidents interactively
"""

import os
import pickle
import pandas as pd
from similarity import find_similar_incidents, analyze_locations
from persistence import load_all_for_search


def load_models():
    """Load saved models and data"""
    # Try new location first, fall back to legacy
    legacy_path = os.path.join('output', 'models', 'models.pkl')
    old_path = 'models.pkl'
    
    models_path = None
    if os.path.exists(legacy_path):
        models_path = legacy_path
    elif os.path.exists(old_path):
        models_path = old_path
    
    if models_path is None:
        print("Error: models.pkl not found!")
        print("Please run main.py first to generate the models.")
        return None, None, None, None, None
    
    print("Loading models and data...")
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    
    tfidf_vectorizer = models['tfidf_vectorizer']
    lsi_model = models['lsi_model']
    lsi_vectors = models['lsi_vectors']
    df = models['df']
    
    print(f"Loaded models for {len(df):,} incidents")
    return tfidf_vectorizer, lsi_model, lsi_vectors, df, models


def interactive_search():
    """Interactive semantic search interface"""
    # Load models
    tfidf_vectorizer, lsi_model, lsi_vectors, df, models = load_models()
    
    if tfidf_vectorizer is None:
        return
    
    print("\n" + "="*80)
    print("SEMANTIC SEARCH INTERFACE")
    print("="*80)
    print("Enter search queries to find semantically similar incidents.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        query = input("Enter your search query (e.g., 'Noise - Residential'): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            print(f"\nSearching for: '{query}'...")
            print("-" * 80)
            
            results = find_similar_incidents(
                query,
                tfidf_vectorizer,
                lsi_model,
                lsi_vectors,
                df,
                top_k=10
            )
            
            if results:
                print(f"\nFound {len(results)} similar incidents:\n")
                for i, result in enumerate(results, 1):
                    print(f"{i}. Similarity: {result['similarity']:.4f}")
                    print(f"   Complaint: {result['complaint_type']}")
                    print(f"   Descriptor: {result['descriptor']}")
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
                    print(f"   Full text: {result['combined_text']}")
                    print()
                
                # Location analysis
                location_stats = analyze_locations(results, top_n=10)
                if location_stats['total'] > 0:
                    print("\n" + "="*80)
                    print("LOCATION ANALYSIS")
                    print("="*80)
                    
                    # Borough statistics
                    if location_stats['borough_counts']:
                        print("\nIncidents by Borough:")
                        sorted_boroughs = sorted(location_stats['borough_counts'].items(), 
                                                 key=lambda x: x[1], reverse=True)
                        for borough, count in sorted_boroughs:
                            print(f"  {borough}: {count}")
                    
                    # Top streets
                    if location_stats['street_counts']:
                        print("\nTop Streets:")
                        for street, count in list(location_stats['street_counts'].items())[:10]:
                            print(f"  {street}: {count}")
                    
                    # Top zip codes
                    if location_stats['zip_counts']:
                        print("\nTop Zip Codes:")
                        for zip_code, count in list(location_stats['zip_counts'].items())[:10]:
                            print(f"  {zip_code}: {count}")
                    
                    print()
            else:
                print("No similar incidents found.")
            
            print("-" * 80)
            print()
            
        except Exception as e:
            print(f"Error during search: {e}\n")


if __name__ == "__main__":
    interactive_search()
