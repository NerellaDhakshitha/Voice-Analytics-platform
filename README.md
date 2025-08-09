# Voice-Analytics-platform
#AI-powered voice-enabled data analytics tool for instant insights, calculations, and visualizations via speech or text.

An AI-powered platform that lets you analyze datasets using voice commands or text input, providing instant calculations, filtering, sorting, and interactive visualizations — all through a modern Streamlit web interface or CLI.
<h3>Features</h3>
<h5>Advanced Voice Processing</h5>
<ul>
  <li>Multi-Engine Speech Recognition with automatic fallback (Google, Sphinx, Wit.ai, Azure)</li>
  <li>Noise Adaptation & Calibration for optimal audio processing</li>
  <li>Continuous Recognition Mode with real-time transcription</li>
  <li>95%+ Intent Recognition Accuracy using advanced NLP</li>
</ul>
<h5>Intelligent Natural Language Understanding</h5>
<ul>
<li>Smart Pattern Matching powered by SpaCy</li>
<li>Fuzzy Column Matching with error tolerance</li>
<li>Context-Aware Query Processing</li>
<li>100+ Supported Query Types (aggregations, filters, comparisons)</li>
</ul>
 <h5>Dynamic Data Visualization</h5>
<ul>
<li>Auto-Chart Selection based on data characteristics</li>
<li>Interactive Plotly Visualizations (bar, line, scatter, heatmaps, box plots)</li>
<li>>Real-time Performance Analytics</li
<li>Smart Default Configurations</li>
</ul>
<h5>Modern Web Interface</h5>
<ul>
<li>Professional Streamlit Dashboard with responsive design</li>
<li>Session Analytics & Tracking</li>
<li>Comprehensive Error Handling</li>
<li>CLI and Web Modes for maximum flexibility</li>
</ul>
<h3>Installation</h3>
<h5>Clone the repository:</h5>
git clone https://github.com/yourusername/vision-assistant.git<br>
cd vision-assistant<br>
<h5>create Virtual Environment</h5>
python -m venv venv<br>
source venv/bin/activate <br>  # macOS/Linux
venv\Scripts\activate <br>     # Windows

<h5>Install dependencies: </h5>
pip install -r requirements.txt<br>
<h5>Set dataset path in config.yaml</h5>
dataset_path: "your_dataset.csv"
<h5>Run the application:</h5>
python main.py<br>

<h3>Usage</h3>
<h5>1. Web Interface (Streamlit)</h5>
streamlit run main.py<br>
Speak or type your commands (e.g., "Show total sales by region as bar chart").
<h5>2.CLI Mode</h5>
python main.py<br>
Speak your query directly (e.g., "Calculate average profit").

<h3>Output</h3>
<h5>Interactive Visualizations</h5>
Generated using Plotly with auto-selected chart types (Bar, Pie, Line, Scatter, Histogram).
Example:
2025-08-09 16:03:45,684 - INFO - ✅ google recognition successful<br>
2025-08-09 16:03:45,700 - INFO - Recognition successful in 5.73s: 'show sales in pie chart'<br>
📝 Processing: 'show sales in pie chart'<br>
2025-08-09 16:03:45,737 - INFO - Command: 'show sales in pie chart' | Intent: show_data | Entities: {'metric': ['Sales'], 'dimension': ['Region'], 'aggregation_type': 'sum', 'chart_type': 'pie', 'filter_column': None, 'filter_value': None, 'sort_column': None, 'sort_order': 'ascending'}<br>
2025-08-09 16:03:45,884 - INFO - Command processed successfully in 0.182s<br>
✅ Generated pie chart showing sum of Sales by Region (⏱️ 0.19s)<br>
<img src="Sample output2.png>

