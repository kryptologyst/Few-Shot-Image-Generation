"""
Sampling script for few-shot image generation models.

This script provides utilities for generating samples from trained models
with various sampling strategies and visualization options.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Dict, Any, Optional, List
import yaml
from omegaconf import OmegaConf
from PIL import Image
import random

from src.models.few_shot_generator import FewShotGenerator, FewShotDiscriminator, MetaLearner
from src.data.dataset import FewShotDataLoader, create_toy_dataset
from src.utils.evaluation import FewShotEvaluator


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FewShotSampler:
    """
    Sampler for few-shot image generation models.
    
    Provides various sampling strategies and visualization utilities.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: torch.device = None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        if os.path.exists(config_path):
            self.config = OmegaConf.load(config_path)
        else:
            self.config = OmegaConf.create({
                'model': {
                    'z_dim': 100,
                    'img_size': 64,
                    'img_channels': 3,
                    'hidden_dim': 512,
                    'num_layers': 4,
                    'use_spectral_norm': True,
                    'use_self_attention': False,
                }
            })
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Evaluation tools
        self.evaluator = FewShotEvaluator(device=self.device)
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model."""
        # Create model architecture
        generator = FewShotGenerator(**self.config.model)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            # PyTorch Lightning checkpoint
            generator.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # Direct model checkpoint
            generator.load_state_dict(checkpoint)
        
        generator.to(self.device)
        return generator
    
    def generate_samples(
        self,
        num_samples: int = 16,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        truncation: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            temperature: Sampling temperature (higher = more diverse)
            truncation: Truncation parameter for noise sampling
            
        Returns:
            Generated images tensor
        """
        if seed is not None:
            set_seed(seed)
        
        with torch.no_grad():
            # Sample noise
            z = torch.randn(num_samples, self.config.model.z_dim, device=self.device)
            
            # Apply truncation
            if truncation < 1.0:
                z = torch.clamp(z, -truncation, truncation)
            
            # Apply temperature
            if temperature != 1.0:
                z = z * temperature
            
            # Generate images
            generated_images = self.model(z)
            
            # Denormalize to [0, 1]
            generated_images = (generated_images + 1) / 2
            generated_images = torch.clamp(generated_images, 0, 1)
        
        return generated_images
    
    def generate_with_adaptation(
        self,
        support_images: torch.Tensor,
        num_samples: int = 16,
        num_adaptation_steps: int = 5,
        adaptation_lr: float = 0.01,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples with few-shot adaptation.
        
        Args:
            support_images: Support images for adaptation
            num_samples: Number of samples to generate
            num_adaptation_steps: Number of adaptation steps
            adaptation_lr: Learning rate for adaptation
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing generated images and adaptation info
        """
        if seed is not None:
            set_seed(seed)
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Adapt model to support images
        adaptation_info = self.model.adapt_to_task(
            support_images.to(self.device),
            num_adaptation_steps=num_adaptation_steps,
            adaptation_lr=adaptation_lr,
        )
        
        # Generate samples
        generated_images = self.generate_samples(num_samples)
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data = original_params[name].data
        
        return {
            'generated_images': generated_images,
            'adaptation_info': adaptation_info,
        }
    
    def interpolate_latent_space(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Interpolate between two points in latent space.
        
        Args:
            z1: First latent point
            z2: Second latent point
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated images
        """
        with torch.no_grad():
            # Create interpolation weights
            alphas = torch.linspace(0, 1, num_steps, device=self.device)
            
            interpolated_images = []
            for alpha in alphas:
                # Linear interpolation
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                # Generate image
                img = self.model(z_interp.unsqueeze(0))
                img = (img + 1) / 2
                img = torch.clamp(img, 0, 1)
                interpolated_images.append(img)
            
            return torch.cat(interpolated_images, dim=0)
    
    def generate_latent_traversal(
        self,
        base_z: torch.Tensor,
        dimension: int,
        range_values: List[float],
    ) -> torch.Tensor:
        """
        Generate latent space traversal along a specific dimension.
        
        Args:
            base_z: Base latent vector
            dimension: Dimension to traverse
            range_values: Values to traverse
            
        Returns:
            Traversal images
        """
        with torch.no_grad():
            traversal_images = []
            
            for value in range_values:
                z = base_z.clone()
                z[0, dimension] = value
                
                img = self.model(z)
                img = (img + 1) / 2
                img = torch.clamp(img, 0, 1)
                traversal_images.append(img)
            
            return torch.cat(traversal_images, dim=0)
    
    def save_samples(
        self,
        images: torch.Tensor,
        output_path: str,
        nrow: int = 4,
        figsize: tuple = (12, 12),
    ) -> None:
        """
        Save generated samples as a grid image.
        
        Args:
            images: Generated images tensor
            output_path: Path to save the image
            nrow: Number of images per row
            figsize: Figure size
        """
        from torchvision.utils import make_grid
        
        # Create grid
        grid = make_grid(images, nrow=nrow, normalize=False)
        
        # Convert to numpy
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        # Save image
        plt.figure(figsize=figsize)
        plt.imshow(grid_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Samples saved to {output_path}")
    
    def save_individual_samples(
        self,
        images: torch.Tensor,
        output_dir: str,
        prefix: str = "sample",
    ) -> None:
        """
        Save individual samples as separate images.
        
        Args:
            images: Generated images tensor
            output_dir: Directory to save images
            prefix: Prefix for filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            # Convert to PIL Image
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            # Save image
            img_path = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
            img_pil.save(img_path)
        
        print(f"Individual samples saved to {output_dir}")


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description='Generate samples from few-shot image generation model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./samples',
                       help='Output directory for generated samples')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--truncation', type=float, default=1.0,
                       help='Truncation parameter')
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual samples as separate images')
    parser.add_argument('--interpolation', action='store_true',
                       help='Generate latent space interpolation')
    parser.add_argument('--traversal', action='store_true',
                       help='Generate latent space traversal')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize sampler
    sampler = FewShotSampler(
        model_path=args.model_path,
        config_path=args.config_path,
    )
    
    # Generate samples
    print("Generating samples...")
    samples = sampler.generate_samples(
        num_samples=args.num_samples,
        seed=args.seed,
        temperature=args.temperature,
        truncation=args.truncation,
    )
    
    # Save samples
    sampler.save_samples(
        samples,
        os.path.join(args.output_dir, "generated_samples.png"),
        nrow=int(np.sqrt(args.num_samples)),
    )
    
    if args.save_individual:
        sampler.save_individual_samples(
            samples,
            os.path.join(args.output_dir, "individual_samples"),
        )
    
    # Generate interpolation if requested
    if args.interpolation:
        print("Generating latent space interpolation...")
        
        # Sample two random points
        z1 = torch.randn(1, sampler.config.model.z_dim, device=sampler.device)
        z2 = torch.randn(1, sampler.config.model.z_dim, device=sampler.device)
        
        # Interpolate
        interp_samples = sampler.interpolate_latent_space(z1, z2, num_steps=10)
        
        # Save interpolation
        sampler.save_samples(
            interp_samples,
            os.path.join(args.output_dir, "latent_interpolation.png"),
            nrow=10,
        )
    
    # Generate traversal if requested
    if args.traversal:
        print("Generating latent space traversal...")
        
        # Sample base point
        base_z = torch.randn(1, sampler.config.model.z_dim, device=sampler.device)
        
        # Traverse first few dimensions
        traversal_images = []
        for dim in range(min(8, sampler.config.model.z_dim)):
            range_values = torch.linspace(-3, 3, 8, device=sampler.device)
            traversal = sampler.generate_latent_traversal(base_z, dim, range_values)
            traversal_images.append(traversal)
        
        # Combine all traversals
        all_traversals = torch.cat(traversal_images, dim=0)
        
        # Save traversal
        sampler.save_samples(
            all_traversals,
            os.path.join(args.output_dir, "latent_traversal.png"),
            nrow=8,
        )
    
    print("Sampling completed!")


if __name__ == "__main__":
    main()
