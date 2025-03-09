"""
Trainer class for model training and adversarial training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import time
import os
from ..models.base_model import BaseModel
from ..attacks.base_attack import BaseAttack

class Trainer:
    """Trainer class for model training and adversarial training."""
    
    def __init__(self,
                 model: BaseModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 attack: Optional[BaseAttack] = None):
        """
        Initialize trainer.
        
        Args:
            model (BaseModel): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            criterion (nn.Module): Loss function
            optimizer (Optimizer): Optimizer
            scheduler (LRScheduler, optional): Learning rate scheduler
            device (str): Device to use for training
            attack (BaseAttack, optional): Attack for adversarial training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer or torch.optim.Adam(model.parameters())
        self.scheduler = scheduler
        self.device = device
        self.attack = attack
        
        self.model.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
    
    def train_epoch(self) -> tuple:
        """
        Train for one epoch.
        
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)
            
            # Generate adversarial examples if attack is specified
            if self.attack is not None:
                data = self.attack.generate(data, target)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> tuple:
        """
        Validate the model.
        
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self,
             num_epochs: int,
             save_dir: str = './checkpoints',
             save_freq: int = 5) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            num_epochs (int): Number of epochs to train
            save_dir (str): Directory to save checkpoints
            save_freq (int): Frequency of saving checkpoints
            
        Returns:
            dict: Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(time.time() - epoch_start_time)
            
            # Print metrics
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.model.save_checkpoint(
                    checkpoint_path,
                    epoch + 1,
                    self.optimizer,
                    self.scheduler,
                    {
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }
                )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(save_dir, 'best_model.pt')
                self.model.save_checkpoint(
                    best_model_path,
                    epoch + 1,
                    self.optimizer,
                    self.scheduler,
                    {
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }
                )
        
        return self.history 