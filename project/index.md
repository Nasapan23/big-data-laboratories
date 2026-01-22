# Semantic Analysis of NYC 311 Service Requests

## 📋 Project Overview

This project performs semantic analysis and clustering on urban incident reports from NYC 311 Service Requests using **TF-IDF** (Term Frequency-Inverse Document Frequency) and **Latent Semantic Indexing (LSI)**. The system enables intelligent search and pattern discovery across over 1.4 million incident reports.

### Key Features

- 🔍 **Semantic Similarity Search**: Find similar incidents using natural language queries
- 📊 **Clustering Analysis**: Automatically group incidents into meaningful clusters
- 🗺️ **Location-Based Analysis**: Identify common incident areas by street, borough, and zip code
- 📈 **Interactive Visualizations**: 2D cluster visualizations using t-SNE
- 🌐 **Web Interface**: Browser-based search interface with Flask
- 💾 **Data Persistence**: Save and load models, embeddings, and processed data

---

## 🏗️ Architecture

The project follows a modular pipeline architecture:

```
Data Loading → Preprocessing → TF-IDF Vectorization → LSI → Clustering → Visualization
```

### Pipeline Steps

1. **Data Loading**: Load and clean CSV data from NYC Open Data
2. **Text Preprocessing**: Lowercase, remove punctuation/digits, remove stopwords
3. **TF-IDF Vectorization**: Convert text to numerical vectors
4. **Latent Semantic Indexing (LSI)**: Reduce dimensionality while preserving semantic meaning
5. **KMeans Clustering**: Group similar incidents into clusters
6. **Visualization**: 2D reduction and cluster visualization
7. **Semantic Search**: Find similar incidents using cosine similarity

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd project
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
```

3. **Activate the virtual environment:**

   **Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt):**
   ```cmd
   venv\Scripts\activate.bat
   ```

   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Download NLTK stopwords (first time only):**
```python
python -c "import nltk; nltk.download('stopwords')"
```

---

## 🚀 Usage

### 1. Run the Complete Pipeline

Process the dataset and generate all models, embeddings, and visualizations:

```bash
python main.py
```

**What it does:**
- Loads data from `dataset.csv`
- Preprocesses text
- Creates TF-IDF vectors
- Applies LSI (150 components)
- Performs KMeans clustering (7 clusters)
- Generates 2D visualization
- Creates HTML analysis report
- Saves all artifacts to `output/` directory

**Expected Output:**
- `output/models/` - Saved models (TF-IDF, LSI, KMeans)
- `output/embeddings/` - TF-IDF and LSI vectors
- `output/data/` - Processed data and cluster assignments
- `output/visualizations/cluster_visualization.png` - Cluster visualization
- `output/reports/analysis_report.html` - Comprehensive HTML report

**Processing Time:** ~10-30 minutes depending on dataset size (1.4M+ records)

---

### 2. Interactive Command-Line Search

Search for similar incidents from the command line:

```bash
python search_interactive.py
```

**Example queries:**
- `Noise - Residential`
- `Illegal Parking`
- `Blocked Driveway`
- `Noise - Street/Sidewalk`

**Features:**
- Shows top 5 similar incidents
- Displays similarity scores
- Includes location information (address, street, borough, zip)
- Location-based statistics (most common streets/boroughs)

---

### 3. Web-Based Search Interface

Launch the Flask web application for browser-based search:

```bash
python web_search.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

**Features:**
- Interactive search interface
- Real-time semantic search
- Location analysis with statistics
- Example query suggestions
- Responsive design

**To stop the server:** Press `Ctrl+C` in the terminal

---

## 📁 Project Structure

```
project/
├── main.py                    # Main pipeline orchestrator
├── data_loading.py            # CSV loading and cleaning
├── preprocessing.py           # Text preprocessing
├── vectorization.py           # TF-IDF vectorization
├── lsi.py                     # Latent Semantic Indexing
├── similarity.py              # Semantic similarity search
├── clustering.py             # KMeans clustering
├── visualization.py           # 2D visualization (t-SNE)
├── reporting.py              # HTML report generation
├── persistence.py            # Save/load models and data
├── search_interactive.py      # Command-line search tool
├── web_search.py             # Flask web interface
├── requirements.txt          # Python dependencies
├── dataset.csv               # Input data (NYC 311 Service Requests)
│
└── output/                   # Generated outputs
    ├── models/               # Saved models (.pkl files)
    ├── embeddings/           # TF-IDF and LSI vectors
    ├── data/                 # Processed data and cluster assignments
    ├── visualizations/       # Cluster visualization PNG
    └── reports/              # HTML analysis report
```

---

## 📊 Results and Visualizations

### Cluster Visualization

The system generates a 2D visualization of incident clusters:

![Cluster Visualization](output/visualizations/cluster_visualization.png)

**Interpretation:**
- Each point represents an incident
- Colors represent different clusters
- Clusters group semantically similar incidents
- Spatial proximity indicates semantic similarity

### Analysis Report

A comprehensive HTML report is generated at:
```
output/reports/analysis_report.html
```

**Contents:**
- Dataset statistics
- Cluster analysis with representative examples
- Top TF-IDF terms per cluster
- Semantic similarity search examples
- Location-based statistics

**To view:** Open the HTML file in any web browser

---

## 🔍 Example Use Cases

### Finding Similar Incidents

**Query:** `"Noise - Residential"`

**Results:** Returns incidents like:
- Noise - Residential - Loud Music/Party
- Noise - Residential - Banging/Pounding
- Noise - Residential - Television

### Location-Based Pattern Discovery

**Query:** `"Illegal Parking"`

**Location Analysis:**
- Most common streets with illegal parking
- Boroughs with highest incidence
- Zip codes with frequent violations

### Cluster Insights

The system identifies 7 main clusters:
1. **Noise - Residential (Banging/Pounding)**
2. **Illegal Parking (Posted Sign Violations)**
3. **Noise - Street/Sidewalk (Loud Music/Party)**
4. **Blocked Driveway (Partial/No Access)**
5. **Illegal Parking (Blocked Hydrant)**
6. **Noise - Residential (Loud Music/Party)**
7. **Illegal Parking (Blocked Sidewalk/Crosswalk)**

---

## 🛠️ Technical Details

### Technologies Used

- **Python 3.8+**
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: TF-IDF, LSI (TruncatedSVD), KMeans, t-SNE
- **nltk**: Text preprocessing (stopwords)
- **matplotlib**: Visualization
- **umap-learn**: Alternative dimensionality reduction (optional)
- **flask**: Web interface
- **scipy**: Sparse matrix operations

### Model Parameters

- **TF-IDF**: `max_features=5000`, `min_df=2`
- **LSI**: `n_components=150` (adjusted dynamically based on features)
- **KMeans**: `n_clusters=7`, `random_state=42`
- **t-SNE**: `perplexity=30`, `n_components=2`

### Performance Optimizations

- Sampling 30,000 records for visualization (for large datasets)
- Sparse matrix storage for TF-IDF vectors
- Model persistence to avoid recomputation
- Efficient cosine similarity computation

---

## 📝 Dataset Information

### Source
NYC Open Data - 311 Service Requests

### Filtering Criteria
- **Agency**: NYPD
- **Time Range**: ≤ 1 year
- **Complaint Types**: Noise (Residential/Street/Commercial), Illegal Parking, Blocked Driveway

### Data Fields
- `Unique Key`: Unique incident identifier
- `Complaint Type`: Category of complaint
- `Descriptor`: Detailed description
- `Created Date`: Incident timestamp
- `Incident Address`: Street address
- `Street Name`: Street name
- `Incident Zip`: Zip code
- `Borough`: NYC borough
- `City`: City name

### Dataset Size
- **Total Records**: ~1.4 million incidents
- **After Filtering**: ~1.4 million valid records

---

## 🎯 Key Achievements

1. ✅ **Scalable Processing**: Handles 1.4M+ records efficiently
2. ✅ **Semantic Understanding**: LSI captures semantic relationships beyond keywords
3. ✅ **Location Intelligence**: Identifies geographic patterns in incidents
4. ✅ **Interactive Tools**: Both CLI and web interfaces for exploration
5. ✅ **Reproducible**: All models and data are saved for future use
6. ✅ **Modular Design**: Clean separation of concerns, easy to extend

---

## 🔧 Troubleshooting

### Common Issues

**1. Models not found error:**
```
Error: models.pkl not found!
```
**Solution:** Run `python main.py` first to generate models

**2. NLTK stopwords not found:**
```
Resource 'stopwords' not found
```
**Solution:** Run `python -c "import nltk; nltk.download('stopwords')"`

**3. Memory errors with large datasets:**
**Solution:** The system automatically samples data for visualization. For very large datasets, reduce `max_samples` in `visualization.py`

**4. Port already in use (web interface):**
```
Address already in use
```
**Solution:** Change the port in `web_search.py` or stop the existing process

---

## 📚 References

- **TF-IDF**: Term Frequency-Inverse Document Frequency for text vectorization
- **LSI**: Latent Semantic Indexing using TruncatedSVD
- **KMeans**: Unsupervised clustering algorithm
- **t-SNE**: t-distributed Stochastic Neighbor Embedding for visualization
- **NYC Open Data**: https://opendata.cityofnewyork.us/

---

## 👤 Author

Big Data Laboratories Project

---

## 📄 License

This project is for educational purposes.

---

## 🎓 Learning Outcomes

This project demonstrates:
- Text preprocessing and normalization
- TF-IDF vectorization for text analysis
- Dimensionality reduction with LSI
- Unsupervised clustering with KMeans
- Semantic similarity search
- Data visualization techniques
- Web application development with Flask
- Large-scale data processing

---

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the generated HTML report for insights
3. Examine the console output for detailed processing information

---

**Last Updated:** 2025
