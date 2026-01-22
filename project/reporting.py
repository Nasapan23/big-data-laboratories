"""
Reporting Module
Generate HTML report with visualizations, statistics, and analysis results
"""

import os
from datetime import datetime


def generate_html_report(cluster_analysis, df, cluster_labels, cluster_viz_path='cluster_visualization.png',
                        similarity_results=None, output_path='analysis_report.html'):
    """
    Generate an HTML report with cluster analysis, statistics, and visualizations.
    
    Args:
        cluster_analysis: Dictionary with cluster analysis results
        df: DataFrame with incident data
        cluster_labels: Cluster assignments
        cluster_viz_path: Path to cluster visualization image
        similarity_results: Optional dictionary with similarity search examples
        output_path: Path to save HTML report
    """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Analysis Report - NYC 311 Service Requests</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background-color: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-box .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-box .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .cluster-card {{
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .cluster-card h3 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .example-list {{
            list-style: none;
            padding: 0;
        }}
        .example-list li {{
            background-color: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            border-left: 3px solid #95a5a6;
        }}
        .terms-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }}
        .term-badge {{
            background-color: #3498db;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        table th, table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        table th {{
            background-color: #3498db;
            color: white;
        }}
        table tr:hover {{
            background-color: #f5f5f5;
        }}
        .similarity-result {{
            background-color: #e8f5e9;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #4caf50;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NLP Analysis Report</h1>
        <p>NYC 311 Service Requests - Semantic Analysis & Clustering</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <!-- Overall Statistics -->
    <div class="section">
        <h2>Dataset Overview</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{len(df):,}</div>
                <div class="stat-label">Total Incidents</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(cluster_analysis)}</div>
                <div class="stat-label">Number of Clusters</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(set(cluster_labels))}</div>
                <div class="stat-label">Unique Clusters</div>
            </div>
        </div>
    </div>
    
    <!-- Cluster Visualization -->
    <div class="section">
        <h2>Cluster Visualization</h2>
        <div class="visualization">
            <img src="{cluster_viz_path}" alt="Cluster Visualization">
        </div>
    </div>
    
    <!-- Cluster Analysis -->
    <div class="section">
        <h2>Cluster Analysis</h2>
"""
    
    # Add cluster details
    for cluster_id, analysis in sorted(cluster_analysis.items()):
        html_content += f"""
        <div class="cluster-card">
            <h3>Cluster {cluster_id} ({analysis['size']:,} incidents)</h3>
            
            <h4>Representative Examples:</h4>
            <ul class="example-list">
"""
        for example in analysis['examples'][:5]:
            html_content += f"                <li>{example['combined_text']}</li>\n"
        
        html_content += "            </ul>\n"
        
        html_content += f"""
            <h4>Top TF-IDF Terms:</h4>
            <div class="terms-list">
"""
        for term, score in analysis.get('top_terms', [])[:10]:
            html_content += f'                <span class="term-badge">{term} ({score:.3f})</span>\n'
        
        html_content += "            </div>\n        </div>\n"
    
    html_content += "    </div>\n"
    
    # Add similarity search results if provided
    if similarity_results:
        html_content += """
    <div class="section">
        <h2>Semantic Similarity Search Examples</h2>
"""
        for query, results in similarity_results.items():
            html_content += f"""
        <h3>Query: "{query}"</h3>
"""
            for i, result in enumerate(results[:5], 1):
                html_content += f"""
        <div class="similarity-result">
            <strong>{i}. Similarity: {result['similarity']:.4f}</strong><br>
            {result['combined_text']}
        </div>
"""
        html_content += "    </div>\n"
    
    # Cluster distribution table
    html_content += """
    <div class="section">
        <h2>Cluster Distribution</h2>
        <table>
            <thead>
                <tr>
                    <th>Cluster ID</th>
                    <th>Number of Incidents</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
    total = len(df)
    for cluster_id in sorted(cluster_analysis.keys()):
        count = cluster_analysis[cluster_id]['size']
        percentage = (count / total) * 100
        html_content += f"""
                <tr>
                    <td>{cluster_id}</td>
                    <td>{count:,}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
"""
    
    html_content += """            </tbody>
        </table>
    </div>
    
    <div class="footer">
        <p>Generated by NLP Pipeline - TF-IDF + LSI + KMeans Clustering</p>
    </div>
</body>
</html>"""
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_path}")

