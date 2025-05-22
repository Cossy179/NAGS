import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
import time
from collections import deque

class CurriculumTrainer:
    """Implements curriculum learning for chess position evaluation and policy learning"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # Training hyperparameters
        self.base_lr = config.get('base_lr', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.batch_size = config.get('batch_size', 1024)
        self.num_epochs = config.get('num_epochs', 100)
        
        # Curriculum stages
        self.curriculum_stages = [
            {'name': 'basic_positions', 'difficulty': 1, 'epochs': 10},
            {'name': 'middlegame_tactics', 'difficulty': 2, 'epochs': 20},
            {'name': 'complex_endgames', 'difficulty': 3, 'epochs': 30},
            {'name': 'grandmaster_games', 'difficulty': 4, 'epochs': 40},
        ]
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.base_lr,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2
        )
        
        # Loss functions
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.CrossEntropyLoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(config.get('log_dir', 'logs'), 'curriculum_training'))
        
        # Training state
        self.current_stage = 0
        self.global_step = 0
        self.best_validation_loss = float('inf')
        
        # Experience replay buffer
        self.replay_buffer_size = config.get('replay_buffer_size', 10000)
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        
        # Cross-validation
        self.cross_val_k = config.get('cross_val_k', 5)
        self.cross_val_fold = 0
        
    def prepare_curriculum_dataset(self, stage):
        """Prepare dataset for the current curriculum stage"""
        stage_info = self.curriculum_stages[stage]
        
        # Load the appropriate dataset based on stage difficulty
        # This is a placeholder - actual implementation would load real datasets
        if stage_info['name'] == 'basic_positions':
            # Dataset with simple positions (e.g., early opening positions)
            return self._load_dataset('basic_positions.pt')
        elif stage_info['name'] == 'middlegame_tactics':
            # Dataset with tactical middlegame positions
            return self._load_dataset('middlegame_tactics.pt')
        elif stage_info['name'] == 'complex_endgames':
            # Dataset with complex endgame positions
            return self._load_dataset('complex_endgames.pt')
        elif stage_info['name'] == 'grandmaster_games':
            # Dataset from high-level GM games
            return self._load_dataset('grandmaster_games.pt')
        else:
            raise ValueError(f"Unknown curriculum stage: {stage_info['name']}")
    
    def _load_dataset(self, filename):
        """Load dataset from file (placeholder function)"""
        # In a real implementation, this would load actual datasets
        # For now, we'll simulate it with random data
        
        # Simulate a dataset with positions, target values, and policies
        class SimulatedDataset:
            def __init__(self, size, difficulty):
                self.size = size
                self.difficulty = difficulty
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Simulate a chess position as a graph
                # This would be replaced with actual data loading
                num_nodes = random.randint(10, 32)  # Number of pieces on board
                
                # Simulate features, edges, values, and policies
                x = torch.randn(num_nodes, 128)
                edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4))
                piece_types = torch.randint(1, 7, (num_nodes,))
                
                # Target value and policy
                target_value = torch.rand(1) * 2 - 1  # Between -1 and 1
                legal_moves = random.randint(20, 40)
                target_policy = torch.zeros(legal_moves)
                target_policy[random.randint(0, legal_moves-1)] = 1.0
                
                return {
                    'x': x,
                    'edge_index': edge_index,
                    'piece_types': piece_types,
                    'target_value': target_value,
                    'target_policy': target_policy,
                    'legal_moves_mask': torch.ones(legal_moves, dtype=torch.bool)
                }
        
        # Simulate different dataset sizes and complexities based on curriculum stage
        difficulty_to_size = {
            1: 5000,
            2: 10000,
            3: 20000,
            4: 50000
        }
        
        # Extract difficulty from filename
        difficulty = 1
        for stage in self.curriculum_stages:
            if stage['name'] in filename:
                difficulty = stage['difficulty']
                break
                
        return SimulatedDataset(difficulty_to_size[difficulty], difficulty)
    
    def train(self):
        """Run the complete curriculum training process"""
        for stage_idx, stage_info in enumerate(self.curriculum_stages):
            self.current_stage = stage_idx
            print(f"Starting curriculum stage {stage_idx+1}/{len(self.curriculum_stages)}: {stage_info['name']}")
            
            # Prepare dataset for this stage
            dataset = self.prepare_curriculum_dataset(stage_idx)
            
            # Split dataset for cross-validation
            dataset_size = len(dataset)
            fold_size = dataset_size // self.cross_val_k
            val_start = self.cross_val_fold * fold_size
            val_end = val_start + fold_size
            
            # Create train/val indices
            train_indices = list(range(0, val_start)) + list(range(val_end, dataset_size))
            val_indices = list(range(val_start, val_end))
            
            # Create data loaders
            train_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(train_indices),
                num_workers=4,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(val_indices),
                num_workers=4,
                pin_memory=True
            )
            
            # Train for specified number of epochs in this stage
            for epoch in range(stage_info['epochs']):
                # Train one epoch
                train_loss = self._train_epoch(train_loader, epoch, stage_idx)
                
                # Validate
                val_loss = self._validate(val_loader, epoch, stage_idx)
                
                # Save checkpoint if this is the best model so far
                if val_loss < self.best_validation_loss:
                    self.best_validation_loss = val_loss
                    self._save_checkpoint(f"best_model_stage_{stage_idx}_epoch_{epoch}.pt")
                
                # Always save latest model
                self._save_checkpoint(f"latest_model_stage_{stage_idx}.pt")
                
                # Update learning rate
                self.scheduler.step()
                
                # Log to console
                print(f"Stage {stage_idx+1}, Epoch {epoch+1}/{stage_info['epochs']}, "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Move to next cross-validation fold for next stage
            self.cross_val_fold = (self.cross_val_fold + 1) % self.cross_val_k
            
            # After completing a stage, perform additional fine-tuning with mixed data
            if stage_idx > 0:
                self._mixed_stage_training(stage_idx)
        
        # Final evaluation on a held-out test set
        self._final_evaluation()
        
        # Close TensorBoard writer
        self.writer.close()
    
    def _train_epoch(self, dataloader, epoch, stage_idx):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_value_loss = 0
        epoch_policy_loss = 0
        
        # Difficulty weight increases with stage
        difficulty_weight = 1.0 + 0.5 * stage_idx
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            batch = self._to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate losses
            value_loss = self.value_loss_fn(outputs['value'], batch['target_value'])
            
            # For policy loss, we need to mask out illegal moves
            policy_logits = outputs['policy']
            legal_moves_mask = batch['legal_moves_mask']
            target_policy = batch['target_policy']
            
            # Apply the mask to the policy logits
            masked_policy_logits = policy_logits[legal_moves_mask]
            
            # Calculate policy loss
            policy_loss = self.policy_loss_fn(masked_policy_logits, target_policy)
            
            # Weight the value loss more heavily in later stages
            value_weight = 1.0 * difficulty_weight
            policy_weight = 1.0
            
            # Combined loss
            loss = value_weight * value_loss + policy_weight * policy_loss
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_value_loss += value_loss.item()
            epoch_policy_loss += policy_loss.item()
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                self.writer.add_scalar(f'Train/Loss/Stage_{stage_idx}/Batch', loss.item(), self.global_step)
                self.writer.add_scalar(f'Train/ValueLoss/Stage_{stage_idx}/Batch', value_loss.item(), self.global_step)
                self.writer.add_scalar(f'Train/PolicyLoss/Stage_{stage_idx}/Batch', policy_loss.item(), self.global_step)
                self.global_step += 1
                
            # Add to replay buffer
            if random.random() < 0.1:  # Sample 10% of batches for replay
                self.replay_buffer.append(batch)
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / len(dataloader)
        avg_value_loss = epoch_value_loss / len(dataloader)
        avg_policy_loss = epoch_policy_loss / len(dataloader)
        
        # Log epoch metrics
        self.writer.add_scalar(f'Train/Loss/Stage_{stage_idx}/Epoch', avg_loss, epoch)
        self.writer.add_scalar(f'Train/ValueLoss/Stage_{stage_idx}/Epoch', avg_value_loss, epoch)
        self.writer.add_scalar(f'Train/PolicyLoss/Stage_{stage_idx}/Epoch', avg_policy_loss, epoch)
        
        # Experience replay - train on random samples from previous stages
        if len(self.replay_buffer) > 0 and stage_idx > 0:
            self._experience_replay()
        
        return avg_loss
    
    def _validate(self, dataloader, epoch, stage_idx):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        val_value_loss = 0
        val_policy_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                batch = self._to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate losses
                value_loss = self.value_loss_fn(outputs['value'], batch['target_value'])
                
                # For policy loss, mask out illegal moves
                policy_logits = outputs['policy']
                legal_moves_mask = batch['legal_moves_mask']
                target_policy = batch['target_policy']
                
                # Apply the mask to the policy logits
                masked_policy_logits = policy_logits[legal_moves_mask]
                
                # Calculate policy loss
                policy_loss = self.policy_loss_fn(masked_policy_logits, target_policy)
                
                # Combined loss (equal weighting for validation)
                loss = value_loss + policy_loss
                
                # Update metrics
                val_loss += loss.item()
                val_value_loss += value_loss.item()
                val_policy_loss += policy_loss.item()
        
        # Calculate average loss
        avg_loss = val_loss / len(dataloader)
        avg_value_loss = val_value_loss / len(dataloader)
        avg_policy_loss = val_policy_loss / len(dataloader)
        
        # Log metrics
        self.writer.add_scalar(f'Validation/Loss/Stage_{stage_idx}', avg_loss, epoch)
        self.writer.add_scalar(f'Validation/ValueLoss/Stage_{stage_idx}', avg_value_loss, epoch)
        self.writer.add_scalar(f'Validation/PolicyLoss/Stage_{stage_idx}', avg_policy_loss, epoch)
        
        return avg_loss
    
    def _experience_replay(self):
        """Train on samples from the replay buffer"""
        if len(self.replay_buffer) == 0:
            return
            
        self.model.train()
        
        # Sample a batch from the replay buffer
        replay_samples = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
        
        # Combine samples into a batch
        batch = self._combine_replay_samples(replay_samples)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch)
        
        # Calculate losses
        value_loss = self.value_loss_fn(outputs['value'], batch['target_value'])
        
        # For policy loss, mask out illegal moves
        policy_logits = outputs['policy']
        legal_moves_mask = batch['legal_moves_mask']
        target_policy = batch['target_policy']
        
        # Apply the mask to the policy logits
        masked_policy_logits = policy_logits[legal_moves_mask]
        
        # Calculate policy loss
        policy_loss = self.policy_loss_fn(masked_policy_logits, target_policy)
        
        # Combined loss with higher weight on older examples
        loss = value_loss + policy_loss
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        # Log replay metrics
        self.writer.add_scalar('Replay/Loss', loss.item(), self.global_step)
    
    def _combine_replay_samples(self, samples):
        """Combine individual samples into a single batch"""
        # This is a placeholder - would need to be implemented for actual data format
        # For demonstration purposes, we'll just return the first sample
        return samples[0]
    
    def _mixed_stage_training(self, current_stage_idx):
        """Train on mixed data from current and previous stages"""
        print(f"Running mixed stage training after stage {current_stage_idx}")
        
        # Load datasets from all completed stages
        datasets = []
        for i in range(current_stage_idx + 1):
            datasets.append(self.prepare_curriculum_dataset(i))
        
        # Create a combined dataset
        # This is a simplified approach - a real implementation would need
        # proper dataset combination logic
        class CombinedDataset(torch.utils.data.Dataset):
            def __init__(self, datasets, weights=None):
                self.datasets = datasets
                self.weights = weights or [1.0] * len(datasets)
                self.dataset_sizes = [len(ds) for ds in datasets]
                self.total_size = sum(self.dataset_sizes)
                
            def __len__(self):
                return self.total_size
                
            def __getitem__(self, idx):
                # Randomly select a dataset based on weights
                dataset_idx = random.choices(
                    range(len(self.datasets)), 
                    weights=self.weights, 
                    k=1
                )[0]
                
                # Get a random sample from the selected dataset
                sample_idx = random.randint(0, len(self.datasets[dataset_idx]) - 1)
                return self.datasets[dataset_idx][sample_idx]
        
        # Create weights that favor the current stage
        weights = [0.5 ** (current_stage_idx - i) for i in range(current_stage_idx + 1)]
        
        # Create combined dataset and loader
        combined_dataset = CombinedDataset(datasets, weights)
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Train for a few epochs on the combined data
        mixed_epochs = 5
        for epoch in range(mixed_epochs):
            # Train one epoch
            train_loss = self._train_epoch(combined_loader, epoch, current_stage_idx)
            
            # Log to console
            print(f"Mixed training after Stage {current_stage_idx}, "
                  f"Epoch {epoch+1}/{mixed_epochs}, "
                  f"Loss: {train_loss:.6f}")
            
        # Save the mixed-trained model
        self._save_checkpoint(f"mixed_trained_stage_{current_stage_idx}.pt")
    
    def _final_evaluation(self):
        """Final evaluation on a held-out test set"""
        print("Running final evaluation...")
        
        # Load test dataset (could be a specific challenging dataset)
        test_dataset = self._load_dataset('test_positions.pt')
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluate
        self.model.eval()
        test_loss = 0
        test_value_loss = 0
        test_policy_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                batch = self._to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate losses
                value_loss = self.value_loss_fn(outputs['value'], batch['target_value'])
                
                # For policy loss, mask out illegal moves
                policy_logits = outputs['policy']
                legal_moves_mask = batch['legal_moves_mask']
                target_policy = batch['target_policy']
                
                # Apply the mask to the policy logits
                masked_policy_logits = policy_logits[legal_moves_mask]
                
                # Calculate policy loss
                policy_loss = self.policy_loss_fn(masked_policy_logits, target_policy)
                
                # Combined loss
                loss = value_loss + policy_loss
                
                # Update metrics
                test_loss += loss.item()
                test_value_loss += value_loss.item()
                test_policy_loss += policy_loss.item()
        
        # Calculate average loss
        avg_loss = test_loss / len(test_loader)
        avg_value_loss = test_value_loss / len(test_loader)
        avg_policy_loss = test_policy_loss / len(test_loader)
        
        # Log metrics
        self.writer.add_scalar('Test/Loss', avg_loss, 0)
        self.writer.add_scalar('Test/ValueLoss', avg_value_loss, 0)
        self.writer.add_scalar('Test/PolicyLoss', avg_policy_loss, 0)
        
        print(f"Final test results - "
              f"Loss: {avg_loss:.6f}, "
              f"Value Loss: {avg_value_loss:.6f}, "
              f"Policy Loss: {avg_policy_loss:.6f}")
    
    def _save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_stage': self.current_stage,
            'global_step': self.global_step,
            'best_validation_loss': self.best_validation_loss,
            'cross_val_fold': self.cross_val_fold,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found.")
            return False
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_stage = checkpoint.get('current_stage', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_validation_loss = checkpoint.get('best_validation_loss', float('inf'))
        self.cross_val_fold = checkpoint.get('cross_val_fold', 0)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return True
    
    def _to_device(self, batch):
        """Move batch data to the specified device"""
        # This is a placeholder - would need to be implemented for actual data format
        # For now, return the batch unchanged
        return batch 