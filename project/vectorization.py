"""
TF-IDF Vectorization Module
Step 3: TF-IDF vectorization using TfidfVectorizer
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def create_tfidf_vectorizer(max_features=5000, min_df=2):
    """
    Create and configure TF-IDF vectorizer.
    
    Args:
        max_features: Maximum number of features (terms) to use
        min_df: Minimum document frequency (ignore terms that appear in fewer documents)
        
    Returns:
        Configured TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        lowercase=False,  # We already preprocessed text to lowercase
        token_pattern=r'\b[a-z]+\b',  # Match words (letters only)
        ngram_range=(1, 1)  # Use unigrams only
    )
    return vectorizer


def vectorize_texts(texts, vectorizer=None, max_features=5000, min_df=2):
    """
    Vectorize texts using TF-IDF.
    
    Args:
        texts: List or Series of preprocessed text strings
        vectorizer: Optional pre-fitted vectorizer (for transforming new data)
        max_features: Maximum number of features (if creating new vectorizer)
        min_df: Minimum document frequency (if creating new vectorizer)
        
    Returns:
        tuple: (tfidf_matrix, vectorizer)
            - tfidf_matrix: Sparse matrix of TF-IDF features
            - vectorizer: Fitted TfidfVectorizer
    """
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer(max_features=max_features, min_df=min_df)
        tfidf_matrix = vectorizer.fit_transform(texts)
    else:
        tfidf_matrix = vectorizer.transform(texts)
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return tfidf_matrix, vectorizer
