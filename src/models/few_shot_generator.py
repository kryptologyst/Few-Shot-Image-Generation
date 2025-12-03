"""
Few-shot Image Generation Models

This module implements various approaches for few-shot image generation including:
- Meta-learning based generators (MAML-style adaptation)
- Transfer learning with pre-trained models
- Conditional GANs for few-shot scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math


class FewShotGenerator(nn.Module):
    """
    Generator for few-shot image generation using meta-learning approach.
    
    This generator can quickly adapt to new image domains with just a few examples
    by using gradient-based meta-learning principles.
    """
    
    def __init__(
        self,
        z_dim: int = 100,
        img_channels: int = 3,
        img_size: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 4,
        use_spectral_norm: bool = True,
        use_self_attention: bool = False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        
        # Build generator layers
        layers = []
        in_dim = z_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim // (2 ** i)
            if i == num_layers - 1:
                out_dim = img_channels * img_size * img_size
            
            linear = nn.Linear(in_dim, out_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(0.2))
            
            in_dim = out_dim
        
        layers.append(nn.Tanh())
        self.generator = nn.Sequential(*layers)
        
        # Self-attention for better quality (optional)
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = SelfAttention(img_channels)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate images from noise.
        
        Args:
            z: Random noise tensor of shape (batch_size, z_dim)
            
        Returns:
            Generated images of shape (batch_size, img_channels, img_size, img_size)
        """
        batch_size = z.size(0)
        
        # Generate through fully connected layers
        x = self.generator(z)
        
        # Reshape to image format
        x = x.view(batch_size, self.img_channels, self.img_size, self.img_size)
        
        # Apply self-attention if enabled
        if self.use_self_attention:
            x = self.attention(x)
        
        return x
    
    def adapt_to_task(
        self, 
        support_images: torch.Tensor, 
        num_adaptation_steps: int = 5,
        adaptation_lr: float = 0.01
    ) -> Dict[str, Any]:
        """
        Adapt the generator to a new task using support images.
        
        Args:
            support_images: Support images for adaptation (batch_size, channels, height, width)
            num_adaptation_steps: Number of gradient steps for adaptation
            adaptation_lr: Learning rate for adaptation
            
        Returns:
            Dictionary containing adaptation information
        """
        original_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Create optimizer for adaptation
        optimizer = torch.optim.Adam(self.parameters(), lr=adaptation_lr)
        
        adaptation_losses = []
        
        for step in range(num_adaptation_steps):
            optimizer.zero_grad()
            
            # Generate random noise
            batch_size = support_images.size(0)
            z = torch.randn(batch_size, self.generator.z_dim, device=support_images.device)
            
            # Generate images
            generated_images = self.forward(z)
            
            # Compute adaptation loss (reconstruction + perceptual)
            recon_loss = F.mse_loss(generated_images, support_images)
            
            # Add perceptual loss using pre-trained features
            perceptual_loss = self._compute_perceptual_loss(generated_images, support_images)
            
            total_loss = recon_loss + 0.1 * perceptual_loss
            total_loss.backward()
            
            optimizer.step()
            adaptation_losses.append(total_loss.item())
        
        return {
            'original_params': original_params,
            'adaptation_losses': adaptation_losses,
            'final_loss': adaptation_losses[-1] if adaptation_losses else 0.0
        }
    
    def _compute_perceptual_loss(
        self, 
        generated: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss using pre-trained features."""
        # Simple L1 loss for now - can be enhanced with VGG features
        return F.l1_loss(generated, target)


class FewShotDiscriminator(nn.Module):
    """
    Discriminator for few-shot image generation.
    
    Uses spectral normalization and progressive growing for stable training.
    """
    
    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 4,
        use_spectral_norm: bool = True,
        use_self_attention: bool = False,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        
        # Build discriminator layers
        layers = []
        
        # Input layer
        in_channels = img_channels
        out_channels = hidden_dim // (2 ** (num_layers - 1))
        
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        layers.append(conv)
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Hidden layers
        for i in range(num_layers - 1):
            in_channels = out_channels
            out_channels = min(hidden_dim, out_channels * 2)
            
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            
            if i < num_layers - 2:  # Don't add batch norm to last layer
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        
        # Final classification layer
        final_linear = nn.Linear(out_channels, 1)
        if use_spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        layers.append(final_linear)
        
        self.discriminator = nn.Sequential(*layers)
        
        # Self-attention (optional)
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = SelfAttention(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate between real and fake images.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            
        Returns:
            Discriminator output of shape (batch_size, 1)
        """
        return self.discriminator(x)


class SelfAttention(nn.Module):
    """Self-attention module for improved image generation quality."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.m = channels // 8
        
        self.f = nn.Conv2d(channels, self.m, kernel_size=1)
        self.g = nn.Conv2d(channels, self.m, kernel_size=1)
        self.h = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        
        # Compute attention maps
        f = self.f(x).view(batch_size, self.m, height * width)
        g = self.g(x).view(batch_size, self.m, height * width)
        h = self.h(x).view(batch_size, channels, height * width)
        
        # Compute attention weights
        attention = torch.bmm(f.transpose(1, 2), g)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(h, attention.transpose(1, 2))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x


class MetaLearner(nn.Module):
    """
    Meta-learning framework for few-shot image generation.
    
    Implements MAML (Model-Agnostic Meta-Learning) for rapid adaptation
    to new image domains with minimal examples.
    """
    
    def __init__(
        self,
        generator: FewShotGenerator,
        discriminator: FewShotDiscriminator,
        meta_lr: float = 0.001,
        adaptation_lr: float = 0.01,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.meta_lr = meta_lr
        self.adaptation_lr = adaptation_lr
        
        # Meta-optimizers
        self.generator_meta_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=meta_lr
        )
        self.discriminator_meta_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=meta_lr
        )
    
    def meta_train_step(
        self,
        support_tasks: List[torch.Tensor],
        query_tasks: List[torch.Tensor],
        num_adaptation_steps: int = 5,
    ) -> Dict[str, float]:
        """
        Perform one meta-training step.
        
        Args:
            support_tasks: List of support image batches for each task
            query_tasks: List of query image batches for each task
            num_adaptation_steps: Number of adaptation steps per task
            
        Returns:
            Dictionary of losses
        """
        task_losses = []
        
        for support_images, query_images in zip(support_tasks, query_tasks):
            # Adapt generator to this task
            gen_adaptation_info = self.generator.adapt_to_task(
                support_images, num_adaptation_steps, self.adaptation_lr
            )
            
            # Compute query loss
            with torch.no_grad():
                batch_size = query_images.size(0)
                z = torch.randn(batch_size, self.generator.z_dim, device=query_images.device)
                generated_query = self.generator(z)
                
                # Generator loss (adversarial + reconstruction)
                real_pred = self.discriminator(query_images)
                fake_pred = self.discriminator(generated_query)
                
                gen_loss = F.binary_cross_entropy_with_logits(
                    fake_pred, torch.ones_like(fake_pred)
                )
                
                recon_loss = F.mse_loss(generated_query, query_images)
                total_gen_loss = gen_loss + 0.1 * recon_loss
                
                task_losses.append(total_gen_loss.item())
        
        # Meta-update
        meta_loss = torch.tensor(sum(task_losses) / len(task_losses))
        meta_loss.backward()
        
        self.generator_meta_optimizer.step()
        self.generator_meta_optimizer.zero_grad()
        
        return {
            'meta_loss': meta_loss.item(),
            'avg_task_loss': sum(task_losses) / len(task_losses),
            'task_losses': task_losses
        }
    
    def generate_samples(
        self, 
        num_samples: int = 16, 
        device: torch.device = None
    ) -> torch.Tensor:
        """Generate samples using the current generator."""
        if device is None:
            device = next(self.generator.parameters()).device
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.generator.z_dim, device=device)
            samples = self.generator(z)
        
        return samples
