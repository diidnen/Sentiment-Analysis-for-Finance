import os
import pandas as pd
import torch
import torch.nn as nn

# Import local modules
from bert_bilstm import BERTBiLSTMModel
from dataset import prepare_dataloaders
from optimizer import get_optimizer
from train import ModelTrainer

def train_bilstm_bert(path, batch_size=16, max_len=128, num_epochs=3):
    '''
    Function to train BiLSTM+BERT model using financial text data.
    
    Args:
        path: Path to data directory
        batch_size: Batch size for training
        max_len: Maximum sequence length
        num_epochs: Number of training epochs
        
    Returns:
        DataFrame with predictions
    '''
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataloaders
    print("Preparing dataloaders...")
    dataloaders = prepare_dataloaders(
        data_file='Bilstmbert/Sentences_66Agree.txt',
        batch_size=batch_size,
        max_len=max_len
    )
    
    # Initialize model
    print("Initializing model...")
    model = BERTBiLSTMModel(
        bert_model_name='bert-base-uncased',
        hidden_size=256,
        num_classes=3,
        dropout_rate=0.2
    )
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Get optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    optimizer, scheduler = get_optimizer(
        model=model,
        train_dataloader=dataloaders['train'],
        num_epochs=num_epochs
    )
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        save_path=os.path.join(path, 'models', 'best_bilstm_bert_model.pt')
    )
    
    # Train the model
    print("Starting training...")
    history = trainer.train(num_epochs=num_epochs)
    
    # Make predictions on test data
    print("Making predictions...")
    predictions = trainer.predict()
    
    # Save predictions
    result = pd.DataFrame(predictions, columns=['negative_prob', 'neutral_prob', 'positive_prob'])
    
    # Add predicted labels
    result['predicted_label'] = result[['negative_prob', 'neutral_prob', 'positive_prob']].idxmax(axis=1)
    result['predicted_label'] = result['predicted_label'].map({
        'negative_prob': 'negative',
        'neutral_prob': 'neutral',
        'positive_prob': 'positive'
    })
    
    # Save results
    print("Saving results...")
    result.to_csv(os.path.join(path, 'predictions.csv'), index=False)
    
    return result

if __name__ == "__main__":
    # Set paths
    data_path = "./Data/"
    
    # Run training
    result_df = train_bilstm_bert(
        path=data_path,
        batch_size=32,
        max_len=256,
        num_epochs=5
    )
    
    print("Training completed successfully!")
    print(f"Results saved to {data_path}predictions.csv")