import torch
import numpy as np
import time
import os
from tqdm import tqdm
import pandas as pd
from optimizer import get_optimizer

class ModelTrainer:
    """Class to handle model training and evaluation"""
    def __init__(self, model, dataloaders, optimizer, scheduler, criterion, device, save_path):
        """
        Initialize the trainer
        
        Args:
            model: The neural network model
            dataloaders: Dictionary with 'train', 'val', and optionally 'test' dataloaders
            optimizer: Model optimizer
            scheduler: Learning rate scheduler
            criterion: Loss function
            device: Device to run the model on ('cuda' or 'cpu')
            save_path: Path to save model checkpoints
        """
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        
        # Create save directory if it doesn't exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Track metrics
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self):
        """Train model for one epoch"""
        self.model.train()
        train_loss = 0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(self.dataloaders['train'], desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with gradient accumulation
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            train_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return train_loss / len(self.dataloaders['train'])
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc='Validating'):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)  # 获取概率最高的类别索引
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(self.dataloaders['val'])
        accuracy = 100 * correct / total
        
        return avg_val_loss, accuracy
    
    def train(self, num_epochs):
        """
        Train the model for specified number of epochs
        
        Args:
            num_epochs: Number of training epochs
            
        Returns:
            History of metrics
        """
        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, accuracy = self.validate()
            
            # Track metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(accuracy)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Model saved to {self.save_path}")
            
            # Print epoch results
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs} - {elapsed_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Accuracy: {accuracy:.2f}%')
            print('-' * 50)
        
        return self.history
    
    def predict(self, dataloader=None):
        """
        Make predictions with the best model
        
        Args:
            dataloader: DataLoader for prediction (uses test dataloader if None)
        
        Returns:
            Numpy array of predictions
        """
        # Use test dataloader if none provided
        if dataloader is None:
            if 'test' not in self.dataloaders:
                raise ValueError("Test dataloader not found")
            dataloader = self.dataloaders['test']
        
        # Load best model
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)