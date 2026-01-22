"""
Web-based Semantic Search Interface
Simple Flask web application for browser-based semantic search
"""

import os
import pickle
from flask import Flask, render_template_string, request, jsonify
from similarity import find_similar_incidents, analyze_locations

app = Flask(__name__)

# Global variables to store models (loaded once at startup)
tfidf_vectorizer = None
lsi_model = None
lsi_vectors = None
df = None


def load_models():
    """Load saved models and data"""
    global tfidf_vectorizer, lsi_model, lsi_vectors, df
    
    # Try new location first, fall back to legacy
    legacy_path = os.path.join('output', 'models', 'models.pkl')
    old_path = 'models.pkl'
    
    models_path = None
    if os.path.exists(legacy_path):
        models_path = legacy_path
    elif os.path.exists(old_path):
        models_path = old_path
    
    if models_path is None:
        print(f"Error: models.pkl not found!")
        print("Please run main.py first to generate the models.")
        return False
    
    print("Loading models and data...")
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    
    tfidf_vectorizer = models['tfidf_vectorizer']
    lsi_model = models['lsi_model']
    lsi_vectors = models['lsi_vectors']
    df = models['df']
    
    print(f"Loaded models for {len(df):,} incidents")
    return True


# HTML template for the search interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Search - NYC 311 Service Requests</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        .header h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .header p {
            color: #666;
        }
        .search-box {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .search-form {
            display: flex;
            gap: 10px;
        }
        .search-input {
            flex: 1;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            outline: none;
            transition: border-color 0.3s;
        }
        .search-input:focus {
            border-color: #667eea;
        }
        .search-button {
            padding: 15px 30px;
            font-size: 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .search-button:hover {
            background: #5568d3;
        }
        .results {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .result-item {
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
            border-radius: 5px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .result-item:hover {
            transform: translateX(5px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .similarity-score {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        .result-text {
            color: #333;
            font-size: 16px;
            line-height: 1.6;
        }
        .result-details {
            margin-top: 10px;
            color: #666;
            font-size: 14px;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .example-queries {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
        .example-queries h3 {
            margin-bottom: 15px;
            color: #333;
        }
        .example-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .example-tag {
            padding: 8px 15px;
            background: #e9ecef;
            border-radius: 20px;
            cursor: pointer;
            transition: background 0.3s;
            font-size: 14px;
        }
        .example-tag:hover {
            background: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Semantic Search</h1>
            <p>NYC 311 Service Requests - Find Similar Incidents</p>
        </div>
        
        <div class="search-box">
            <form class="search-form" onsubmit="search(event)">
                <input 
                    type="text" 
                    class="search-input" 
                    id="query" 
                    placeholder="Enter your search query (e.g., 'Noise - Residential')"
                    value="{{ query if query else '' }}"
                >
                <button type="submit" class="search-button">Search</button>
            </form>
            
            <div class="example-queries">
                <h3>Example Queries:</h3>
                <div class="example-tags">
                    <span class="example-tag" onclick="setQuery('Noise - Residential')">Noise - Residential</span>
                    <span class="example-tag" onclick="setQuery('Illegal Parking - Blocked Driveway')">Illegal Parking</span>
                    <span class="example-tag" onclick="setQuery('Noise - Street/Sidewalk')">Noise - Street/Sidewalk</span>
                    <span class="example-tag" onclick="setQuery('Blocked Driveway')">Blocked Driveway</span>
                </div>
            </div>
        </div>
        
        <div class="results" id="results">
            {% if results %}
                {% if results|length > 0 %}
                    {% for result in results %}
                    <div class="result-item">
                        <div class="result-header">
                            <span class="similarity-score">Similarity: {{ "%.4f"|format(result.similarity) }}</span>
                        </div>
                        <div class="result-text">{{ result.combined_text }}</div>
                        <div class="result-details">
                            <strong>Complaint Type:</strong> {{ result.complaint_type }} | 
                            <strong>Descriptor:</strong> {{ result.descriptor }}
                            {% if result.unique_key %} | <strong>ID:</strong> {{ result.unique_key }}{% endif %}
                            {% if result.created_date %} | <strong>Date:</strong> {{ result.created_date }}{% endif %}
                        </div>
                        {% if result.incident_address or result.street_name or result.borough or result.incident_zip %}
                        <div class="result-location">
                            {% if result.incident_address %}📍 {{ result.incident_address }}{% endif %}
                            {% if result.street_name %}{% if result.incident_address %} | {% endif %}🛣️ {{ result.street_name }}{% endif %}
                            {% if result.borough %}{% if result.incident_address or result.street_name %} | {% endif %}🏙️ {{ result.borough }}{% endif %}
                            {% if result.incident_zip %}{% if result.incident_address or result.street_name or result.borough %} | {% endif %}📮 {{ result.incident_zip }}{% endif %}
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                    
                    {% if location_stats and location_stats.total > 0 %}
                    <div class="location-analysis">
                        <h3>📍 Location Analysis</h3>
                        <div class="location-stats">
                            {% if location_stats.borough_counts_sorted %}
                            <div class="location-section">
                                <h4>Boroughs</h4>
                                <ul class="location-list">
                                    {% for borough, count in location_stats.borough_counts_sorted %}
                                    <li>{{ borough }} <span class="location-count">{{ count }}</span></li>
                                    {% endfor %}
                                </ul>
                            </div>
                            {% endif %}
                            
                            {% if location_stats.street_counts_list %}
                            <div class="location-section">
                                <h4>Top Streets</h4>
                                <ul class="location-list">
                                    {% for street, count in location_stats.street_counts_list %}
                                    <li>{{ street }} <span class="location-count">{{ count }}</span></li>
                                    {% endfor %}
                                </ul>
                            </div>
                            {% endif %}
                            
                            {% if location_stats.zip_counts_list %}
                            <div class="location-section">
                                <h4>Top Zip Codes</h4>
                                <ul class="location-list">
                                    {% for zip_code, count in location_stats.zip_counts_list %}
                                    <li>{{ zip_code }} <span class="location-count">{{ count }}</span></li>
                                    {% endfor %}
                                </ul>
                            </div>
                            {% endif %}
                        </div>
                    </div>
                    {% endif %}
                {% else %}
                    <div class="no-results">
                        <p>No results found. Try a different query.</p>
                    </div>
                {% endif %}
            {% else %}
                <div class="no-results">
                    <p>Enter a search query above to find similar incidents.</p>
                </div>
            {% endif %}
        </div>
    </div>
    
    <script>
        function setQuery(query) {
            document.getElementById('query').value = query;
            document.getElementById('query').focus();
        }
        
        function search(event) {
            event.preventDefault();
            const query = document.getElementById('query').value.trim();
            if (!query) return;
            
            document.getElementById('results').innerHTML = '<div class="loading">Searching...</div>';
            
            window.location.href = '/?q=' + encodeURIComponent(query);
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main search page"""
    query = request.args.get('q', '').strip()
    results = []
    location_stats = None
    
    if query and tfidf_vectorizer is not None:
        try:
            results = find_similar_incidents(
                query,
                tfidf_vectorizer,
                lsi_model,
                lsi_vectors,
                df,
                top_k=20
            )
            if results:
                location_stats = analyze_locations(results, top_n=10)
                # Convert dictionaries to sorted lists for Jinja2 template
                if location_stats:
                    if location_stats.get('borough_counts'):
                        location_stats['borough_counts_sorted'] = sorted(
                            location_stats['borough_counts'].items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )
                    # Convert street_counts and zip_counts to lists for Jinja2
                    if location_stats.get('street_counts'):
                        location_stats['street_counts_list'] = list(location_stats['street_counts'].items())
                    if location_stats.get('zip_counts'):
                        location_stats['zip_counts_list'] = list(location_stats['zip_counts'].items())
        except Exception as e:
            print(f"Error during search: {e}")
            results = []
    
    return render_template_string(HTML_TEMPLATE, query=query, results=results, location_stats=location_stats)


@app.route('/api/search')
def api_search():
    """API endpoint for programmatic access"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    
    if tfidf_vectorizer is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        results = find_similar_incidents(
            query,
            tfidf_vectorizer,
            lsi_model,
            lsi_vectors,
            df,
            top_k=20
        )
        return jsonify({'query': query, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*80)
    print("Starting Semantic Search Web Interface")
    print("="*80)
    
    if not load_models():
        print("\nCannot start web server without models.")
        print("Please run 'python main.py' first to generate models.pkl")
        exit(1)
    
    print("\nWeb interface ready!")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
