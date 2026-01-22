You are working on a Python NLP project that performs semantic analysis and clustering on urban incident reports using TF-IDF and Latent Semantic Indexing (LSI).

📂 Dataset

Input is a CSV file exported from NYC Open Data (311 Service Requests).

The dataset has been filtered to include:

Agency: NYPD

A limited time range (≤ 1 year)

Complaint Types such as:

Noise – Residential

Noise – Street/Sidewalk

Noise – Commercial

Illegal Parking

Blocked Driveway

📄 Relevant CSV columns

The CSV contains (at minimum):

Unique Key

Complaint Type

Descriptor

Created Date

🧩 Text field to analyze

Create a single text field by concatenating:

Complaint Type + " - " + Descriptor


This combined field represents the incident description and is the only text used for NLP.

🎯 Project Objectives

Implement the following pipeline step by step:

1️⃣ Data Loading

Load the CSV using pandas

Drop rows where Descriptor is missing or empty

Keep only the columns needed for analysis

2️⃣ Text Preprocessing

For the combined text field:

Convert to lowercase

Remove punctuation and digits

Remove English stopwords

Keep preprocessing lightweight (no deep NLP, no transformers)

3️⃣ TF-IDF Vectorization

Use TfidfVectorizer

Set reasonable limits (max_features, min_df)

Build a TF-IDF matrix for all incident descriptions

4️⃣ Latent Semantic Indexing (LSI)

Apply TruncatedSVD on the TF-IDF matrix

Choose a fixed number of components (e.g. 100–200)

Store the reduced semantic representation

5️⃣ Semantic Similarity Search

Implement a function that:

Takes a new incident text as input

Applies the same preprocessing and TF-IDF + LSI transformations

Computes cosine similarity

Returns the top-k most semantically similar incidents

6️⃣ Clustering in Semantic Space

Apply KMeans clustering on the LSI vectors

Choose a reasonable number of clusters (e.g. 5–10)

Assign each incident to a cluster

For each cluster:

Print representative examples

Extract and display top TF-IDF terms

7️⃣ Visualization

Reduce LSI vectors to 2D using UMAP or t-SNE

Plot a scatter diagram colored by cluster

Keep visuals simple and readable

🛠️ Technical Constraints

Language: Python

Libraries:

pandas

numpy

scikit-learn

nltk

matplotlib

umap-learn (optional, for visualization)

❌ Do NOT use:

Deep learning models

Transformers (BERT, etc.)

Spark / Hadoop

Databases or web frameworks

📦 Expected Output

Clean, modular Python code

Reproducible results

Clear separation between:

preprocessing

vectorization

semantic modeling

clustering

visualization

The final result should demonstrate that LSI improves semantic similarity and grouping of urban incident reports compared to raw keyword matching.