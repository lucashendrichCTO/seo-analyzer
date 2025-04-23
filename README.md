# SEO Content Analyzer

A powerful web application that analyzes content for SEO optimization using advanced NLP and LLM techniques. The analyzer provides comprehensive insights including:

- Word count and keyword density analysis
- Meta tags and heading structure evaluation
- LLM-based attention mapping
- Semantic relevance scoring
- Google Trends integration
- Intelligent SEO suggestions

## Features

### Basic Analysis
- Word count and content structure analysis
- Keyword density and frequency metrics
- Meta tags evaluation
- Heading structure analysis
- Real-time content processing
- Support for both HTML and plain text content

### Advanced LLM Features
- BERT-based attention mapping to identify important content sections
- Semantic similarity analysis between content and keywords
- Intelligent keyword relevance scoring
- LLM-powered readability assessment
- Content structure recommendations

### Google Trends Integration
- Real-time keyword trend analysis
- Historical trend data visualization
- Trending direction indicators
- Relative interest scores

### User Interface
- Clean, modern UI using Tailwind CSS
- Real-time analysis updates
- Responsive design for all devices
- Interactive visualizations
- Detailed tooltips and explanations

## Dependencies

### Core Requirements
- Python 3.8+
- Flask 2.3.3
- BeautifulSoup4 4.12.2
- NLTK 3.8.1
- Requests 2.31.0
- PyTrends 4.9.2
- Waitress 2.1.2 (Production server)
- Python-dotenv 1.0.0

### LLM Components
- PyTorch 2.7.0
- Transformers 4.51.3
- NumPy 1.24.3

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd seo-analyzer
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words'); nltk.download('stopwords')"
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter your content in one of two ways:
   - Paste HTML content
   - Enter plain text content

4. Click "Analyze" to get comprehensive SEO insights

## Analysis Results

The analyzer provides detailed metrics including:

### Content Analysis
- Word count and content length assessment
- Keyword density and distribution
- Heading structure analysis
- Meta tags evaluation

### LLM Insights
- Attention scores for content sections
- Semantic relevance scores for keywords
- Content readability metrics
- Structure recommendations

### Trend Analysis
- Google Trends integration results
- Keyword popularity metrics
- Trending directions
- Historical interest data

### Recommendations
- SEO improvement suggestions
- Content structure recommendations
- Keyword optimization tips
- Meta tag suggestions

## Development

### Project Structure
```
seo-analyzer/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── static/            # Static assets (CSS, JS)
└── templates/         # HTML templates
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BERT model provided by Hugging Face Transformers
- Google Trends integration via PyTrends
- NLTK for natural language processing
