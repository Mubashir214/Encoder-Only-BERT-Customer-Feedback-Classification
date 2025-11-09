import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    try:
        model_path = "./bert-feedback-best"
        
        # Check if model files exist
        required_files = [
            'pytorch_model.bin', 'config.json', 'vocab.txt',
            'tokenizer_config.json', 'special_tokens_map.json'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
        
        if missing_files:
            st.error(f"Missing model files: {', '.join(missing_files)}")
            return None, None
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_sentiment(texts, tokenizer, model, top_k=3):
    """
    Predict sentiment for a list of texts
    """
    try:
        # Get label mapping from model config
        id2label = model.config.id2label
        
        # Tokenize inputs
        encodings = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        
        results = []
        for i, text in enumerate(texts):
            pred_idx = probabilities[i].argmax()
            pred_label = id2label.get(int(pred_idx), "unknown")
            pred_prob = float(probabilities[i, pred_idx])
            
            # Get top-k predictions
            top_indices = probabilities[i].argsort()[-top_k:][::-1]
            topk_predictions = [
                (id2label.get(int(idx), "unknown"), float(probabilities[i, idx]))
                for idx in top_indices
            ]
            
            results.append({
                "text": text,
                "predicted_label": pred_label,
                "confidence": pred_prob,
                "top_predictions": topk_predictions
            })
        
        return results
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return []

def main():
    st.title("🎯 BERT Sentiment Analysis App")
    st.write("Analyze sentiment in customer feedback using your trained BERT model")
    
    # Load model
    with st.spinner("Loading model and tokenizer..."):
        tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("""
        ❌ Failed to load model. Please ensure all model files are present in the `bert-feedback-best` folder:
        
        Required files:
        - `pytorch_model.bin` (model weights)
        - `config.json` (model configuration)  
        - `vocab.txt` (tokenizer vocabulary)
        - `tokenizer_config.json` (tokenizer settings)
        - `special_tokens_map.json` (special tokens)
        """)
        return
    
    # Display model info in sidebar
    st.sidebar.title("Model Information")
    st.sidebar.write(f"**Model Type:** {model.config.model_type}")
    st.sidebar.write(f"**Number of Labels:** {model.config.num_labels}")
    st.sidebar.write(f"**ID to Label Mapping:**")
    st.sidebar.json(model.config.id2label)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Single Text Analysis", "Batch Analysis", "Model Info"]
    )
    
    if app_mode == "Single Text Analysis":
        st.header("🔍 Single Text Analysis")
        
        # Text input
        user_input = st.text_area(
            "Enter text to analyze:",
            height=100,
            placeholder="Type your text here... (e.g., 'The product is amazing and delivery was fast!')"
        )
        
        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            show_top_k = st.checkbox("Show top 3 predictions", value=True)
        with col2:
            show_confidence = st.checkbox("Show confidence scores", value=True)
        
        if st.button("Analyze Sentiment") and user_input:
            with st.spinner("Analyzing sentiment..."):
                results = predict_sentiment([user_input], tokenizer, model)
                
                if results:
                    result = results[0]
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Color code based on sentiment
                    sentiment_color = {
                        "positive": "🟢",
                        "negative": "🔴", 
                        "neutral": "🟡",
                        "unknown": "⚪"
                    }
                    
                    emoji = sentiment_color.get(result["predicted_label"].lower(), "⚪")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="Predicted Sentiment",
                            value=f"{emoji} {result['predicted_label'].upper()}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Confidence",
                            value=f"{result['confidence']:.2%}"
                        )
                    
                    # Confidence bar
                    st.progress(float(result['confidence']))
                    
                    if show_top_k:
                        st.subheader("Top Predictions")
                        for label, prob in result["top_predictions"]:
                            emoji = sentiment_color.get(label.lower(), "⚪")
                            st.write(f"{emoji} **{label.upper()}**: {prob:.2%}")
    
    elif app_mode == "Batch Analysis":
        st.header("📊 Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file with a 'text' column",
            type=['csv'],
            help="Your CSV should have a column named 'text' containing the reviews to analyze"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column")
                    return
                
                if st.button("Analyze All Reviews"):
                    with st.spinner("Analyzing batch data..."):
                        # Analyze in batches to avoid memory issues
                        batch_size = 32
                        all_results = []
                        
                        progress_bar = st.progress(0)
                        total_batches = (len(df) + batch_size - 1) // batch_size
                        
                        for i in range(0, len(df), batch_size):
                            batch_texts = df['text'].iloc[i:i+batch_size].fillna('').astype(str).tolist()
                            batch_results = predict_sentiment(batch_texts, tokenizer, model)
                            all_results.extend(batch_results)
                            
                            # Update progress
                            progress = min((i + batch_size) / len(df), 1.0)
                            progress_bar.progress(progress)
                        
                        if all_results:
                            # Add results to dataframe
                            results_df = df.copy()
                            results_df['predicted_sentiment'] = [r['predicted_label'] for r in all_results]
                            results_df['confidence'] = [r['confidence'] for r in all_results]
                            
                            # Display results
                            st.subheader("Analysis Results")
                            st.dataframe(results_df[['text', 'predicted_sentiment', 'confidence']].head(10))
                            
                            # Summary statistics
                            st.subheader("📈 Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            
                            sentiment_counts = results_df['predicted_sentiment'].value_counts()
                            
                            with col1:
                                st.metric("Total Reviews", len(results_df))
                            with col2:
                                st.metric("Positive Reviews", 
                                        sentiment_counts.get('positive', 0))
                            with col3:
                                st.metric("Average Confidence", 
                                        f"{results_df['confidence'].mean():.2%}")
                            
                            # Visualization
                            st.subheader("Sentiment Distribution")
                            fig, ax = plt.subplots()
                            sentiment_counts.plot(kind='bar', ax=ax, color=['red', 'gray', 'green'])
                            ax.set_title('Sentiment Distribution')
                            ax.set_ylabel('Count')
                            ax.tick_params(axis='x', rotation=45)
                            st.pyplot(fig)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif app_mode == "Model Info":
        st.header("🤖 Model Information")
        
        st.subheader("Model Configuration")
        st.json(model.config.to_dict())
        
        st.subheader("File Check")
        model_path = "./bert-feedback-best"
        files_status = {}
        for file in ['pytorch_model.bin', 'config.json', 'vocab.txt', 
                     'tokenizer_config.json', 'special_tokens_map.json']:
            file_path = os.path.join(model_path, file)
            files_status[file] = "✅ Found" if os.path.exists(file_path) else "❌ Missing"
        
        for file, status in files_status.items():
            st.write(f"{status} - `{file}`")
        
        st.subheader("Example Predictions")
        example_texts = [
            "The product is excellent and delivery was super fast!",
            "The item was okay, nothing special.",
            "Terrible quality and poor customer service."
        ]
        
        for text in example_texts:
            if st.button(f"Test: '{text}'"):
                results = predict_sentiment([text], tokenizer, model)
                if results:
                    result = results[0]
                    st.write(f"**Prediction:** {result['predicted_label']} (Confidence: {result['confidence']:.2%})")

if __name__ == "__main__":
    main()