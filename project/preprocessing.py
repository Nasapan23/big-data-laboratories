"""
Text Preprocessing Module
Step 2: Lightweight text preprocessing (lowercase, remove punctuation/digits, stopwords)
"""

import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Get English stopwords
STOP_WORDS = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Preprocess text for NLP analysis.
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and digits (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    
    # Join words back
    text = ' '.join(words)
    
    return text


def preprocess_dataframe(df, text_column='combined_text'):
    """
    Preprocess text column in a DataFrame.
    
    Args:
        df: DataFrame with text column
        text_column: Name of the text column to preprocess
        
    Returns:
        DataFrame with preprocessed text in a new column 'preprocessed_text'
    """
    df = df.copy()
    df['preprocessed_text'] = df[text_column].apply(preprocess_text)
    return df
