# Sentiment Analysis for Financial Texts using BiLSTM & BERT

This project applies **Natural Language Processing (NLP)** techniques to analyze sentiment in financial texts. It leverages **BERT** for feature extraction and a **BiLSTM classifier** to predict sentiment as **positive**, **neutral**, or **negative**.

## Key Features

- Processed financial text data (**Sentences_66Agree.txt**) into a **BERT-compatible format** (tokenization, attention masks). Applied **text tokenization**, **padding**, and **encoding** for NLP model training.  
- **Model Architecture**:  
  - **BERT + BiLSTM**: Used **BERT embeddings** as input to a **BiLSTM model**. Extracted **first** and **last time steps** from BiLSTM, combined them, and applied a **sigmoid activation** for classification.
- Identified **class imbalance** and applied **oversampling & undersampling** techniques to balance sentiment distribution.
- Implemented **multi-threaded training** for faster convergence.

## Tech Stack

- **Python**, **PyTorch**, **Hugging Face Transformers**  
- **BERT**, **BiLSTM**, **Adam Optimizer**  
- **NLP Preprocessing**, **Tokenization**, **OneCycleLR**

## Possible Next Steps

- Explore **Transformer-based classification** (**RoBERTa**, **FinBERT**)  
- Fine-tune **BERT** on **domain-specific financial datasets**  
- Implement **explainability methods** (**SHAP**, **LIME**) for model interpretation

## Setup & Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Sentiment-Analysis-for-Finance.git
    ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the training script:
    ```bash
    python train.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
