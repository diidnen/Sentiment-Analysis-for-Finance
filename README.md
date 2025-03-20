Sentiment Analysis for Financial Texts using BiLSTM & BERT
This project applies Natural Language Processing (NLP) techniques to analyze sentiment in financial texts. It leverages BERT for feature extraction and a BiLSTM classifier to predict sentiment as positive, neutral, or negative.

Processed financial text data (Sentences_66Agree.txt) into a BERT-compatible format (<CLS> tokenization, attention masks).
Applied text tokenization, padding, and encoding for NLP model training.
Model Architecture:

BERT + BiLSTM: Used BERT embeddings as input to a BiLSTM model.
Extracted first and last time steps from BiLSTM, combined them, and applied a sigmoid activation for classification.

Identified class imbalance and applied oversampling & undersampling techniques to balance sentiment distribution.
Implemented multi-threaded training for faster convergence.

 Python, PyTorch, Hugging Face Transformers
 BERT, BiLSTM, Adam Optimizer
 NLP Preprocessing, Tokenization, OneCycleLR

Possible Next Steps:

Explore Transformer-based classification (RoBERTa, FinBERT)
Fine-tune BERT on domain-specific financial datasets
Implement explainability methods (SHAP, LIME) for model interpretation
