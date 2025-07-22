# Legal Document Sentiment Analyzer

**AI-powered legal document analysis for compliance and risk assessment**

## Overview

The Legal Document Sentiment Analyzer is an AI-powered tool designed to help legal professionals quickly assess sentiment and risk levels in legal documents. By combining state-of-the-art NLP models with domain-specific legal analysis, this application provides:

- 📄 **Document sentiment classification** (Positive/Negative/Neutral)
- ⚖️ **Legal-specific sentiment analysis**
- ⚠️ **Risk assessment** (High/Medium/Low)
- 📊 **Interactive visualizations** for batch processing
- 📈 **Confidence scoring** for each analysis

Built with Python and Gradio, this tool helps legal teams identify potential risks, ensure compliance, and prioritize document review.

## Key Features

### Advanced NLP Models
- **Dual-model architecture**:
  - `cardiffnlp/twitter-roberta-base-sentiment` for general sentiment analysis
  - `ProsusAI/finbert` for legal/financial domain analysis
- Graceful degradation with fallback models
- GPU acceleration support

### Comprehensive Analysis
- Sentiment classification with confidence scoring
- Legal-specific sentiment evaluation
- Custom risk assessment logic:
  ```python
  if primary_sent == 'negative' and legal_sent == 'negative':
      if primary_conf > 0.8 and legal_conf > 0.8:
          return 'high'
      elif primary_conf > 0.6 or legal_conf > 0.6:
          return 'medium'
  ```

### Interactive Dashboard
- 4-panel visualization dashboard:
  1. Sentiment distribution (pie chart)
  2. Risk level analysis (bar chart)
  3. Confidence scores (histogram)
  4. Document risk matrix (scatter plot)
- Batch processing of multiple documents
- Exportable results in DataFrame format

### User-Friendly Interface
- Tab-based organization (single vs. batch analysis)
- Sample documents for quick testing
- Clear risk indicators and summaries
- Mobile-responsive design

## Technical Stack

### Core Technologies
- **Python 3.8+**
- **PyTorch** (model inference)
- **Hugging Face Transformers** (NLP models)
- **Gradio** (web interface)
- **Plotly** (interactive visualizations)
- **Pandas** (data processing)

### NLP Models
| Model | Type | Purpose |
|-------|------|---------|
| `cardiffnlp/twitter-roberta-base-sentiment` | RoBERTa-base | Primary sentiment analysis |
| `ProsusAI/finbert` | BERT-based | Legal/financial sentiment |
| `nlptown/bert-base-multilingual-uncased-sentiment` | BERT | Fallback model |

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- NVIDIA GPU (recommended for faster inference)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/maihun-rsc/legal-sentiment-analysis-initial.git
   cd legal-sentiment-analyzer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   python app.py
   ```

5. Access the interface at `http://localhost:7860` in your browser

## Usage

### Single Document Analysis
1. Navigate to the "Single Document Analysis" tab
2. Paste legal text into the input box
3. Click "Analyze Document"
4. Review results:
   - Primary and legal sentiment
   - Confidence scores
   - Risk assessment
   - Summary recommendation

### Batch Document Analysis
1. Navigate to the "Batch Document Analysis" tab
2. Enter multiple documents separated by blank lines
3. Click "Analyze All Documents"
4. Explore:
   - Statistical summary
   - Detailed results table
   - Interactive dashboard
   - Risk prioritization

![Interface Tabs](https://via.placeholder.com/600x300?text=Single+and+Batch+Analysis+Tabs)

## Use Cases

- **Contract review**: Identify potentially problematic clauses
- **Compliance monitoring**: Detect non-compliant language
- **Litigation risk assessment**: Prioritize high-risk documents
- **Legal opinion analysis**: Classify sentiment in counsel opinions
- **Client communication review**: Analyze sentiment in correspondence

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## Acknowledgements

- Hugging Face for transformer models and libraries
- Cardiff NLP for the Twitter-RoBERTa model
- Prosus AI for FinBERT
- Gradio team for the excellent UI framework

---

**Disclaimer**: This tool is designed to assist legal professionals but does not replace legal advice. Always consult with qualified counsel for important decisions.
