import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import transformers for sentiment analysis
import torch
import io
import base64
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import subprocess
import sys

# Ensure we have the latest transformers version
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers"])
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print("✅ Transformers upgraded successfully")
except Exception as e:
    print(f"⚠️ Transformers upgrade failed: {e}")

# Set up the sentiment analysis models
print("Loading sentiment analysis models...")

# Helper function to mimic pipeline behavior
def predict_sentiment(text, tokenizer, model):
    """Predict sentiment for text using given model"""
    if not text or not text.strip():
        return []
    
    device = next(model.parameters()).device
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    scores = torch.softmax(outputs.logits, dim=-1)[0]
    return [{"label": model.config.id2label[i], "score": score.item()} 
            for i, score in enumerate(scores)]

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load RoBERTa/BERT model
try:
    print("Loading RoBERTa model...")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer1 = AutoTokenizer.from_pretrained(model_name)
    model1 = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model1.eval()
    print("✅ RoBERTa sentiment model loaded")
    
    # Create a wrapper function that handles device and empty text
    def roberta_pipeline(text):
        if not text.strip():
            return []
        return predict_sentiment(text, tokenizer1, model1)
        
    sentiment_pipeline = roberta_pipeline
    
except Exception as e:
    print(f"Error loading RoBERTa: {e}")
    print("Loading BERT fallback model...")
    try:
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer1 = AutoTokenizer.from_pretrained(model_name)
        model1 = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        model1.eval()
        
        def bert_pipeline(text):
            if not text.strip():
                return []
            return predict_sentiment(text, tokenizer1, model1)
            
        sentiment_pipeline = bert_pipeline
        print("✅ BERT sentiment model loaded")
    except Exception as e2:
        print(f"Critical error: {e2}")
        # Create dummy pipeline to prevent complete failure
        def dummy_pipeline(text):
            return [{"label": "NEUTRAL", "score": 1.0}]
        sentiment_pipeline = dummy_pipeline
        print("⚠️ Using dummy sentiment model")

# Load FinBERT model
try:
    print("Loading FinBERT model...")
    legal_model_name = "ProsusAI/finbert"
    tokenizer2 = AutoTokenizer.from_pretrained(legal_model_name)
    model2 = AutoModelForSequenceClassification.from_pretrained(legal_model_name).to(device)
    model2.eval()
    print("✅ FinBERT legal model loaded")
    
    def finbert_pipeline(text):
        if not text.strip():
            return []
        return predict_sentiment(text, tokenizer2, model2)
        
    legal_sentiment_pipeline = finbert_pipeline
    
except Exception as e:
    print(f"Error loading FinBERT: {e}")
    print("Using fallback model for legal analysis")
    legal_sentiment_pipeline = sentiment_pipeline

class LegalSentimentAnalyzer:
    def __init__(self):
        self.label_mapping = {
            'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive',
            'NEGATIVE': 'negative', 'NEUTRAL': 'neutral', 'POSITIVE': 'positive',
            'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive',
            '0 stars': 'negative', '1 star': 'negative', '2 stars': 'neutral',
            '3 stars': 'neutral', '4 stars': 'positive', '5 stars': 'positive'
        }

    def preprocess_text(self, text):
        """Clean and preprocess legal text"""
        if not text or pd.isna(text):
            return ""

        text = ' '.join(text.split())
        if len(text) > 512:
            text = text[:509] + "..."
        return text

    def analyze_single_document(self, text):
        """Analyze a single document"""
        text = self.preprocess_text(text)

        if not text:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'legal_sentiment': 'neutral',
                'legal_confidence': 0.0,
                'risk_level': 'low',
                'summary': 'Empty document'
            }

        # Primary sentiment
        try:
            primary_result = sentiment_pipeline(text)
            if not primary_result:
                primary_label, primary_score = 'neutral', 0.5
            else:
                primary_sentiment = max(primary_result, key=lambda x: x['score'])
                primary_label = self.label_mapping.get(
                    primary_sentiment['label'],
                    primary_sentiment['label'].lower()
                )
                primary_score = primary_sentiment['score']
        except Exception as e:
            print(f"Primary sentiment error: {e}")
            primary_label, primary_score = 'neutral', 0.5

        # Legal sentiment
        try:
            legal_result = legal_sentiment_pipeline(text)
            if not legal_result:
                legal_label, legal_score = 'neutral', 0.5
            else:
                legal_sentiment = max(legal_result, key=lambda x: x['score'])
                legal_label = self.label_mapping.get(
                    legal_sentiment['label'],
                    legal_sentiment['label'].lower()
                )
                legal_score = legal_sentiment['score']
        except Exception as e:
            print(f"Legal sentiment error: {e}")
            legal_label, legal_score = 'neutral', 0.5

        # Risk assessment
        risk_level = self.assess_risk(primary_label, legal_label, primary_score, legal_score)

        # Generate summary
        summary = self.generate_summary(primary_label, legal_label, risk_level, primary_score)

        return {
            'sentiment': primary_label,
            'confidence': round(primary_score, 3),
            'legal_sentiment': legal_label,
            'legal_confidence': round(legal_score, 3),
            'risk_level': risk_level,
            'summary': summary
        }

    def assess_risk(self, primary_sent, legal_sent, primary_conf, legal_conf):
        """Assess legal risk"""
        if primary_sent == 'negative' and legal_sent == 'negative':
            if primary_conf > 0.8 and legal_conf > 0.8:
                return 'high'
            elif primary_conf > 0.6 or legal_conf > 0.6:
                return 'medium'
        elif primary_sent == 'negative' or legal_sent == 'negative':
            return 'medium'
        return 'low'

    def generate_summary(self, primary_sent, legal_sent, risk_level, confidence):
        """Generate analysis summary"""
        summaries = {
            'high': f"⚠️ HIGH RISK: Document shows {primary_sent} sentiment with {confidence:.1%} confidence. Requires immediate legal review.",
            'medium': f"⚡ MEDIUM RISK: Document has {primary_sent} sentiment. Consider legal consultation.",
            'low': f"✅ LOW RISK: Document shows {primary_sent} sentiment with acceptable risk level."
        }
        return summaries.get(risk_level, "Analysis completed.")

    def analyze_multiple_documents(self, documents_text):
        """Analyze multiple documents from text input"""
        if not documents_text:
            return "Please enter documents to analyze.", None, None

        # Split documents by double newlines or numbered list
        documents = []
        lines = documents_text.strip().split('\n')
        current_doc = ""

        for line in lines:
            line = line.strip()
            if not line:
                if current_doc:
                    documents.append(current_doc)
                    current_doc = ""
            else:
                current_doc += " " + line

        if current_doc:
            documents.append(current_doc)

        if not documents:
            return "No valid documents found.", None, None

        # Analyze each document
        results = []
        for i, doc in enumerate(documents):
            result = self.analyze_single_document(doc)
            result['document_id'] = i + 1
            result['text_preview'] = doc[:100] + "..." if len(doc) > 100 else doc
            results.append(result)

        # Create results DataFrame
        df = pd.DataFrame(results)

        # Generate summary stats
        summary_text = self.generate_batch_summary(df)

        # Create visualizations
        fig = self.create_visualizations(df)

        return summary_text, df, fig

    def generate_batch_summary(self, df):
        """Generate summary for batch analysis"""
        total_docs = len(df)
        sentiment_counts = df['sentiment'].value_counts()
        risk_counts = df['risk_level'].value_counts()

        high_risk = risk_counts.get('high', 0)
        medium_risk = risk_counts.get('medium', 0)
        low_risk = risk_counts.get('low', 0)

        negative_docs = sentiment_counts.get('negative', 0)
        positive_docs = sentiment_counts.get('positive', 0)
        neutral_docs = sentiment_counts.get('neutral', 0)

        avg_confidence = df['confidence'].mean()

        summary = f"""
📊 **LEGAL DOCUMENT ANALYSIS SUMMARY**
═══════════════════════════════════════

📋 **Total Documents Analyzed:** {total_docs}

🎯 **Sentiment Distribution:**
• Positive: {positive_docs} ({positive_docs/total_docs*100:.1f}%)
• Negative: {negative_docs} ({negative_docs/total_docs*100:.1f}%)
• Neutral: {neutral_docs} ({neutral_docs/total_docs*100:.1f}%)

⚠️ **Risk Assessment:**
• High Risk: {high_risk} documents ({high_risk/total_docs*100:.1f}%)
• Medium Risk: {medium_risk} documents ({medium_risk/total_docs*100:.1f}%)
• Low Risk: {low_risk} documents ({low_risk/total_docs*100:.1f}%)

📈 **Average Confidence:** {avg_confidence:.1%}

🔍 **Recommendations:**
{"• IMMEDIATE ACTION: Review high-risk documents" if high_risk > 0 else "• No immediate action required"}
{"• MONITOR: Significant negative sentiment detected" if negative_docs > total_docs * 0.3 else "• Sentiment levels acceptable"}
• Regular monitoring recommended for ongoing compliance
        """

        return summary

    def create_visualizations(self, df):
        """Create interactive visualizations"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Risk Level Analysis',
                          'Confidence Scores', 'Document Risk Matrix'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )

        # 1. Sentiment Distribution (Pie Chart)
        sentiment_counts = df['sentiment'].value_counts()
        colors = ['#ff4444', '#ffaa00', '#44ff44']

        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                   name="Sentiment", marker_colors=colors),
            row=1, col=1
        )

        # 2. Risk Level Distribution (Bar Chart)
        risk_counts = df['risk_level'].value_counts()
        risk_colors = {'high': '#ff0000', 'medium': '#ff8800', 'low': '#00ff00'}

        fig.add_trace(
            go.Bar(x=risk_counts.index, y=risk_counts.values,
                   name="Risk Level",
                   marker_color=[risk_colors.get(x, '#cccccc') for x in risk_counts.index]),
            row=1, col=2
        )

        # 3. Confidence Score Distribution (Histogram)
        fig.add_trace(
            go.Histogram(x=df['confidence'], nbinsx=20, name="Confidence",
                        marker_color='skyblue'),
            row=2, col=1
        )

        # 4. Risk Matrix (Scatter Plot)
        risk_colors_scatter = {'high': 'red', 'medium': 'orange', 'low': 'green'}

        fig.add_trace(
            go.Scatter(x=df['confidence'], y=df['legal_confidence'],
                      mode='markers', name="Documents",
                      marker=dict(
                          size=10,
                          color=[risk_colors_scatter.get(x, 'gray') for x in df['risk_level']],
                          line=dict(width=1, color='black')
                      ),
                      text=df['document_id'],
                      textposition="middle center"),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Legal Document Sentiment Analysis Dashboard",
            showlegend=False,
            height=600,
            template="plotly_white"
        )

        return fig

# Initialize analyzer
analyzer = LegalSentimentAnalyzer()

# Define Gradio interface functions
def analyze_single_text(text):
    """Analyze single document"""
    if not text.strip():
        return "Please enter some text to analyze.", "", "", "", "", ""

    result = analyzer.analyze_single_document(text)

    # Format output
    sentiment = f"🎯 **Primary Sentiment:** {result['sentiment'].title()}"
    confidence = f"📊 **Confidence:** {result['confidence']:.1%}"
    legal_sentiment = f"⚖️ **Legal Sentiment:** {result['legal_sentiment'].title()}"
    legal_confidence = f"📈 **Legal Confidence:** {result['legal_confidence']:.1%}"
    risk_level = f"⚠️ **Risk Level:** {result['risk_level'].title()}"
    summary = f"📋 **Summary:** {result['summary']}"

    return sentiment, confidence, legal_sentiment, legal_confidence, risk_level, summary

def analyze_batch_text(documents_text):
    """Analyze multiple documents"""
    if not documents_text.strip():
        return "Please enter documents to analyze.", None, None

    summary, df, fig = analyzer.analyze_multiple_documents(documents_text)

    return summary, df, fig

def load_sample_documents():
    """Load sample legal documents"""
    sample_docs = """The contract clearly states that all parties must comply with the terms and conditions as outlined in Section 3.1. Failure to do so may result in legal action.

The plaintiff alleges that the defendant breached the agreement by failing to deliver the goods on time, causing significant financial losses.

This settlement agreement is reached amicably between both parties, with no admission of liability or wrongdoing.

The court finds the defendant liable for damages in the amount of $50,000 due to negligent conduct.

Both parties agree to the terms of this contract and acknowledge their understanding of all provisions contained herein.

The evidence presented clearly demonstrates a pattern of fraudulent behavior by the defendant company.

This legal opinion concludes that the proposed action is within the bounds of applicable law and regulation.

The arbitration clause in the contract is valid and enforceable under state law.

The investigation revealed serious violations of corporate governance policies and procedures.

The settlement provides fair compensation to all affected parties without prolonged litigation."""

    return sample_docs

# Create Gradio interface
with gr.Blocks(title="Legal Document Sentiment Analysis", theme=gr.themes.Soft()) as demo:

    # Header
    gr.Markdown("""
    # �️ Legal Document Sentiment Analysis

    **AI-Powered Legal Text Analysis for Compliance and Risk Assessment**

    This tool uses advanced NLP models (RoBERTa + FinBERT) to analyze legal documents for:
    - Sentiment classification (Positive/Negative/Neutral)
    - Legal risk assessment (High/Medium/Low)
    - Confidence scoring and compliance recommendations
    """)

    # Single Document Analysis Tab
    with gr.Tab("📄 Single Document Analysis"):
        gr.Markdown("### Analyze Individual Legal Document")

        with gr.Row():
            with gr.Column(scale=2):
                single_input = gr.Textbox(
                    lines=8,
                    placeholder="Enter legal document text here...",
                    label="Legal Document Text"
                )
                analyze_btn = gr.Button("🔍 Analyze Document", variant="primary")

            with gr.Column(scale=2):
                sentiment_output = gr.Textbox(label="Primary Sentiment", interactive=False)
                confidence_output = gr.Textbox(label="Confidence Score", interactive=False)
                legal_sentiment_output = gr.Textbox(label="Legal Sentiment", interactive=False)
                legal_confidence_output = gr.Textbox(label="Legal Confidence", interactive=False)
                risk_output = gr.Textbox(label="Risk Level", interactive=False)
                summary_output = gr.Textbox(label="Analysis Summary", interactive=False)

        analyze_btn.click(
            analyze_single_text,
            inputs=[single_input],
            outputs=[sentiment_output, confidence_output, legal_sentiment_output,
                    legal_confidence_output, risk_output, summary_output]
        )

    # Batch Analysis Tab
    with gr.Tab("📊 Batch Document Analysis"):
        gr.Markdown("### Analyze Multiple Documents")
        gr.Markdown("*Enter multiple documents separated by empty lines*")

        with gr.Row():
            with gr.Column():
                batch_input = gr.Textbox(
                    lines=15,
                    placeholder="Enter multiple legal documents here, separated by empty lines...",
                    label="Multiple Legal Documents"
                )

                with gr.Row():
                    batch_analyze_btn = gr.Button("📊 Analyze All Documents", variant="primary")
                    sample_btn = gr.Button("📋 Load Sample Documents", variant="secondary")

        batch_summary = gr.Textbox(
            label="Analysis Summary",
            lines=15,
            interactive=False
        )

        batch_results = gr.Dataframe(
            label="Detailed Results",
            headers=["Document ID", "Text Preview", "Sentiment", "Confidence",
                    "Legal Sentiment", "Legal Confidence", "Risk Level"],
            interactive=False
        )

        batch_plot = gr.Plot(label="Analysis Dashboard")

        batch_analyze_btn.click(
            analyze_batch_text,
            inputs=[batch_input],
            outputs=[batch_summary, batch_results, batch_plot]
        )

        sample_btn.click(
            load_sample_documents,
            outputs=[batch_input]
        )

    # About Tab
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## 🔬 Technical Details

        **Models Used:**
        - **Primary Sentiment**: RoBERTa (Twitter-based sentiment analysis)
        - **Legal Analysis**: FinBERT (Financial/Legal domain-specific model)
        - **Fallback**: BERT Multilingual model

        **Risk Assessment Criteria:**
        - **High Risk**: Negative sentiment in both models with high confidence
        - **Medium Risk**: Negative sentiment in one model or moderate confidence
        - **Low Risk**: Positive/neutral sentiment with acceptable confidence

        **Features:**
        - Real-time sentiment analysis
        - Legal-specific risk assessment
        - Batch processing capabilities
        - Interactive visualizations
        - Exportable results

        **Use Cases:**
        - Contract review and analysis
        - Legal document compliance checking
        - Risk assessment for legal proceedings
        - Client feedback sentiment analysis
        - Legal opinion classification

        ---

        **Created for Legal Professionals and Compliance Teams**

        *This tool is designed to assist legal professionals in analyzing document sentiment and risk levels. Always consult with qualified legal counsel for important decisions.*
        """)

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True
    )