"""
Training script for few-shot image generation models.

This script provides comprehensive training with proper logging, checkpointing,
and evaluation for few-shot image generation using meta-learning approaches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import numpy as np
import random
import os
import argparse
from typing import Dict, Any, Optional
import yaml
from omegaconf import OmegaConf
import wandb

from src.models.few_shot_generator import FewShotGenerator, FewShotDiscriminator, MetaLearner
from src.data.dataset import FewShotDataLoader, create_toy_dataset
from src.utils.evaluation import FewShotEvaluator


class FewShotImageGenerationModule(pl.LightningModule):
    """
    PyTorch Lightning module for few-shot image generation.
    
    Implements meta-learning training with proper logging and evaluation.
    """
    
    def __init__(
        self,
        z_dim: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_dim: int = 512,
        num_layers: int = 4,
        use_spectral_norm: bool = True,
        use_self_attention: bool = False,
        meta_lr: float = 0.001,
        adaptation_lr: float = 0.01,
        num_adaptation_steps: int = 5,
        lambda_recon: float = 0.1,
        lambda_perceptual: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize models
        self.generator = FewShotGenerator(
            z_dim=z_dim,
            img_channels=img_channels,
            img_size=img_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_spectral_norm=use_spectral_norm,
            use_self_attention=use_self_attention,
        )
        
        self.discriminator = FewShotDiscriminator(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_spectral_norm=use_spectral_norm,
            use_self_attention=use_self_attention,
        )
        
        self.meta_learner = MetaLearner(
            generator=self.generator,
            discriminator=self.discriminator,
            meta_lr=meta_lr,
            adaptation_lr=adaptation_lr,
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.MSELoss()
        
        # Training parameters
        self.num_adaptation_steps = num_adaptation_steps
        self.lambda_recon = lambda_recon
        self.lambda_perceptual = lambda_perceptual
        
        # Evaluation
        self.evaluator = FewShotEvaluator(device=self.device)
        
        # Metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise."""
        return self.generator(z)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for meta-learning."""
        support_images = batch['support_images']
        query_images = batch['query_images']
        
        # Meta-learning training
        meta_losses = self.meta_learner.meta_train_step(
            support_tasks=[support_images],
            query_tasks=[query_images],
            num_adaptation_steps=self.num_adaptation_steps,
        )
        
        # Log losses
        self.log('train/meta_loss', meta_losses['meta_loss'], on_step=True, on_epoch=True)
        self.log('train/avg_task_loss', meta_losses['avg_task_loss'], on_step=True, on_epoch=True)
        
        # Store outputs for epoch-end processing
        self.training_step_outputs.append(meta_losses)
        
        return meta_losses['meta_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation step with evaluation metrics."""
        support_images = batch['support_images']
        query_images = batch['query_images']
        
        # Generate samples
        batch_size = support_images.size(0)
        z = torch.randn(batch_size, self.hparams.z_dim, device=self.device)
        generated_images = self.generator(z)
        
        # Compute losses
        real_pred = self.discriminator(query_images)
        fake_pred = self.discriminator(generated_images)
        
        # Generator loss
        gen_loss = self.adversarial_loss(fake_pred, torch.ones_like(fake_pred))
        recon_loss = self.reconstruction_loss(generated_images, query_images)
        total_gen_loss = gen_loss + self.lambda_recon * recon_loss
        
        # Discriminator loss
        real_loss = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        disc_loss = (real_loss + fake_loss) / 2
        
        # Compute evaluation metrics
        metrics = {}
        if batch_idx == 0:  # Only compute expensive metrics on first batch
            try:
                metrics['fid'] = self.evaluator.compute_fid(query_images, generated_images)
                metrics['lpips'] = self.evaluator.compute_lpips(query_images, generated_images)
            except Exception as e:
                print(f"Error computing metrics: {e}")
                metrics['fid'] = 0.0
                metrics['lpips'] = 0.0
        
        # Log metrics
        self.log('val/gen_loss', total_gen_loss, on_step=False, on_epoch=True)
        self.log('val/disc_loss', disc_loss, on_step=False, on_epoch=True)
        self.log('val/recon_loss', recon_loss, on_step=False, on_epoch=True)
        
        for key, value in metrics.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True)
        
        # Store outputs
        self.validation_step_outputs.append({
            'gen_loss': total_gen_loss.item(),
            'disc_loss': disc_loss.item(),
            'recon_loss': recon_loss.item(),
            **metrics
        })
        
        return metrics
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        if self.training_step_outputs:
            avg_meta_loss = np.mean([output['meta_loss'] for output in self.training_step_outputs])
            avg_task_loss = np.mean([output['avg_task_loss'] for output in self.training_step_outputs])
            
            self.log('train/epoch_meta_loss', avg_meta_loss)
            self.log('train/epoch_task_loss', avg_task_loss)
            
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if self.validation_step_outputs:
            avg_gen_loss = np.mean([output['gen_loss'] for output in self.validation_step_outputs])
            avg_disc_loss = np.mean([output['disc_loss'] for output in self.validation_step_outputs])
            avg_recon_loss = np.mean([output['recon_loss'] for output in self.validation_step_outputs])
            
            self.log('val/epoch_gen_loss', avg_gen_loss)
            self.log('val/epoch_disc_loss', avg_disc_loss)
            self.log('val/epoch_recon_loss', avg_recon_loss)
            
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers."""
        gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.meta_lr,
            betas=(0.5, 0.999)
        )
        
        disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.meta_lr,
            betas=(0.5, 0.999)
        )
        
        return [gen_optimizer, disc_optimizer]
    
    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """Generate samples for visualization."""
        with torch.no_grad():
            z = torch.randn(num_samples, self.hparams.z_dim, device=self.device)
            samples = self.generator(z)
        return samples


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def create_callbacks(config: Dict[str, Any]) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='few-shot-gen-{epoch:02d}-{val/fid:.2f}',
        monitor='val/fid',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=config['training']['save_every_n_epochs'],
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val/fid',
        mode='min',
        patience=config['training']['early_stopping_patience'],
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def create_logger(config: Dict[str, Any]) -> Optional[pl.loggers.Logger]:
    """Create logger for training."""
    logger_type = config['logging']['logger_type']
    
    if logger_type == 'tensorboard':
        return TensorBoardLogger(
            save_dir=config['logging']['log_dir'],
            name='few-shot-image-generation',
            version=None,
        )
    elif logger_type == 'wandb':
        wandb.init(
            project=config['logging']['project_name'],
            config=config,
            name=f"few-shot-gen-{config['training']['max_epochs']}epochs",
        )
        return WandbLogger(
            project=config['logging']['project_name'],
            name=f"few-shot-gen-{config['training']['max_epochs']}epochs",
        )
    else:
        return None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train few-shot image generation model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
    else:
        # Default configuration
        config = OmegaConf.create({
            'model': {
                'z_dim': 100,
                'img_size': 64,
                'img_channels': 3,
                'hidden_dim': 512,
                'num_layers': 4,
                'use_spectral_norm': True,
                'use_self_attention': False,
                'meta_lr': 0.001,
                'adaptation_lr': 0.01,
                'num_adaptation_steps': 5,
                'lambda_recon': 0.1,
                'lambda_perceptual': 0.1,
            },
            'data': {
                'dataset_type': 'cifar10',
                'img_size': 64,
                'batch_size': 4,
                'num_workers': 4,
                'num_shots': 5,
                'num_query': 15,
            },
            'training': {
                'max_epochs': 100,
                'checkpoint_dir': './checkpoints',
                'save_every_n_epochs': 10,
                'early_stopping_patience': 20,
            },
            'logging': {
                'logger_type': 'tensorboard',
                'log_dir': './logs',
                'project_name': 'few-shot-image-generation',
            }
        })
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)
    
    # Create toy dataset if needed
    if config.data.dataset_type == 'custom' and not os.path.exists(args.data_path):
        print("Creating toy dataset...")
        create_toy_dataset(
            output_path=args.data_path,
            num_classes=5,
            samples_per_class=100,
            img_size=config.data.img_size,
        )
    
    # Create data loaders
    data_loader = FewShotDataLoader(
        data_path=args.data_path,
        img_size=config.data.img_size,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        dataset_type=config.data.dataset_type,
        num_shots=config.data.num_shots,
        num_query=config.data.num_query,
    )
    
    # Create model
    model = FewShotImageGenerationModule(**config.model)
    
    # Create callbacks and logger
    callbacks = create_callbacks(config)
    logger = create_logger(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',
        devices='auto',
        precision=16,  # Mixed precision
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
    )
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=data_loader.get_train_loader(),
        val_dataloaders=data_loader.get_val_loader(),
        ckpt_path=args.resume_from_checkpoint,
    )
    
    # Final evaluation
    print("Running final evaluation...")
    final_metrics = model.evaluator.evaluate_model(
        model=model,
        data_loader=data_loader.get_val_loader(),
        num_samples=500,
        save_samples=True,
        output_dir=os.path.join(args.output_dir, 'final_evaluation'),
    )
    
    print("Training completed!")
    print(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
