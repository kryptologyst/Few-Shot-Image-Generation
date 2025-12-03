"""
Evaluation metrics for few-shot image generation.

This module provides comprehensive evaluation metrics including:
- FID (Fréchet Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Precision and Recall for generative models
- Few-shot specific metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import lpips
from clean_fid import fid
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import cdist


class InceptionFeatureExtractor(nn.Module):
    """Extract features using pre-trained Inception v3 model."""
    
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained Inception v3
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.eval()
        self.inception.to(self.device)
        
        # Remove final classification layer
        self.inception.fc = nn.Identity()
        
        # Disable gradients
        for param in self.inception.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Input images of shape (batch_size, 3, height, width)
            
        Returns:
            Features of shape (batch_size, 2048)
        """
        # Resize to 299x299 for Inception v3
        if x.size(-1) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1] if needed
        if x.min() >= 0:
            x = x * 2.0 - 1.0
        
        with torch.no_grad():
            features = self.inception(x)
        
        return features


class FewShotEvaluator:
    """
    Comprehensive evaluator for few-shot image generation models.
    
    Provides multiple evaluation metrics including FID, LPIPS, precision/recall,
    and few-shot specific metrics.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        cache_dir: str = "./cache",
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize feature extractors
        self.inception_extractor = InceptionFeatureExtractor(self.device)
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        
        # Cache for precomputed features
        self.feature_cache = {}
    
    def compute_fid(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        batch_size: int = 50,
    ) -> float:
        """
        Compute Fréchet Inception Distance (FID).
        
        Args:
            real_images: Real images tensor
            generated_images: Generated images tensor
            batch_size: Batch size for feature extraction
            
        Returns:
            FID score (lower is better)
        """
        # Extract features
        real_features = self._extract_features_batch(real_images, batch_size)
        gen_features = self._extract_features_batch(generated_images, batch_size)
        
        # Compute FID
        fid_score = self._compute_fid_from_features(real_features, gen_features)
        
        return fid_score
    
    def compute_lpips(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
    ) -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity).
        
        Args:
            real_images: Real images tensor
            generated_images: Generated images tensor
            
        Returns:
            LPIPS score (lower is better)
        """
        # Ensure images are in [-1, 1] range
        real_images = self._normalize_images(real_images)
        generated_images = self._normalize_images(generated_images)
        
        # Compute LPIPS
        with torch.no_grad():
            lpips_scores = self.lpips_model(real_images, generated_images)
        
        return lpips_scores.mean().item()
    
    def compute_precision_recall(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        k: int = 3,
    ) -> Dict[str, float]:
        """
        Compute Precision and Recall for generative models.
        
        Args:
            real_images: Real images tensor
            generated_images: Generated images tensor
            k: Number of nearest neighbors for precision/recall computation
            
        Returns:
            Dictionary containing precision and recall scores
        """
        # Extract features
        real_features = self._extract_features_batch(real_images, batch_size=50)
        gen_features = self._extract_features_batch(generated_images, batch_size=50)
        
        # Compute pairwise distances
        real_distances = cdist(real_features, real_features, metric='euclidean')
        gen_distances = cdist(gen_features, gen_features, metric='euclidean')
        cross_distances = cdist(gen_features, real_features, metric='euclidean')
        
        # Compute precision
        precision_scores = []
        for i in range(len(gen_features)):
            # Find k nearest real images for each generated image
            nearest_real_indices = np.argsort(cross_distances[i])[:k]
            # Check if any of these are also in the k nearest generated images
            nearest_gen_indices = np.argsort(gen_distances[i])[1:k+1]  # Skip self
            precision = len(set(nearest_real_indices) & set(nearest_gen_indices)) / k
            precision_scores.append(precision)
        
        # Compute recall
        recall_scores = []
        for i in range(len(real_features)):
            # Find k nearest generated images for each real image
            nearest_gen_indices = np.argsort(cross_distances[:, i])[:k]
            # Check if any of these are also in the k nearest real images
            nearest_real_indices = np.argsort(real_distances[i])[1:k+1]  # Skip self
            recall = len(set(nearest_gen_indices) & set(nearest_real_indices)) / k
            recall_scores.append(recall)
        
        return {
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'f1': 2 * np.mean(precision_scores) * np.mean(recall_scores) / 
                  (np.mean(precision_scores) + np.mean(recall_scores))
        }
    
    def compute_few_shot_metrics(
        self,
        support_images: torch.Tensor,
        query_images: torch.Tensor,
        generated_images: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute few-shot specific evaluation metrics.
        
        Args:
            support_images: Support images used for adaptation
            query_images: Query images (ground truth)
            generated_images: Generated images
            
        Returns:
            Dictionary containing few-shot metrics
        """
        # Extract features
        support_features = self._extract_features_batch(support_images, batch_size=50)
        query_features = self._extract_features_batch(query_images, batch_size=50)
        gen_features = self._extract_features_batch(generated_images, batch_size=50)
        
        # Compute adaptation quality (how well generated images match support)
        support_gen_distances = cdist(gen_features, support_features, metric='euclidean')
        adaptation_quality = np.mean(np.min(support_gen_distances, axis=1))
        
        # Compute generation quality (how well generated images match query)
        query_gen_distances = cdist(gen_features, query_features, metric='euclidean')
        generation_quality = np.mean(np.min(query_gen_distances, axis=1))
        
        # Compute diversity (how diverse are the generated images)
        gen_gen_distances = cdist(gen_features, gen_features, metric='euclidean')
        # Remove diagonal (self-distances)
        gen_gen_distances = gen_gen_distances[~np.eye(gen_gen_distances.shape[0], dtype=bool)]
        diversity = np.mean(gen_gen_distances)
        
        return {
            'adaptation_quality': adaptation_quality,
            'generation_quality': generation_quality,
            'diversity': diversity,
            'adaptation_generation_ratio': adaptation_quality / generation_quality
        }
    
    def evaluate_model(
        self,
        model: nn.Module,
        data_loader: Any,
        num_samples: int = 1000,
        save_samples: bool = True,
        output_dir: str = "./evaluation_results",
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a few-shot image generation model.
        
        Args:
            model: The model to evaluate
            data_loader: Data loader for evaluation
            num_samples: Number of samples to generate for evaluation
            save_samples: Whether to save generated samples
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_real_images = []
        all_generated_images = []
        all_support_images = []
        all_query_images = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx * data_loader.batch_size >= num_samples:
                    break
                
                support_images = batch['support_images'].to(self.device)
                query_images = batch['query_images'].to(self.device)
                
                # Generate images
                batch_size = support_images.size(0)
                z = torch.randn(batch_size, model.generator.z_dim, device=self.device)
                generated_images = model.generator(z)
                
                all_real_images.append(query_images.cpu())
                all_generated_images.append(generated_images.cpu())
                all_support_images.append(support_images.cpu())
                all_query_images.append(query_images.cpu())
        
        # Concatenate all batches
        real_images = torch.cat(all_real_images, dim=0)[:num_samples]
        generated_images = torch.cat(all_generated_images, dim=0)[:num_samples]
        support_images = torch.cat(all_support_images, dim=0)[:num_samples]
        query_images = torch.cat(all_query_images, dim=0)[:num_samples]
        
        # Compute metrics
        metrics = {}
        
        # Basic metrics
        metrics['fid'] = self.compute_fid(real_images, generated_images)
        metrics['lpips'] = self.compute_lpips(real_images, generated_images)
        
        # Precision/Recall
        pr_metrics = self.compute_precision_recall(real_images, generated_images)
        metrics.update(pr_metrics)
        
        # Few-shot specific metrics
        fs_metrics = self.compute_few_shot_metrics(
            support_images, query_images, generated_images
        )
        metrics.update(fs_metrics)
        
        # Save samples if requested
        if save_samples:
            self._save_evaluation_samples(
                real_images, generated_images, support_images, output_dir
            )
        
        # Save metrics
        self._save_metrics(metrics, output_dir)
        
        return metrics
    
    def _extract_features_batch(
        self, 
        images: torch.Tensor, 
        batch_size: int = 50
    ) -> np.ndarray:
        """Extract features in batches to avoid memory issues."""
        features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            batch_features = self.inception_extractor(batch)
            features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def _compute_fid_from_features(
        self, 
        real_features: np.ndarray, 
        gen_features: np.ndarray
    ) -> float:
        """Compute FID from pre-extracted features."""
        # Compute means and covariances
        real_mean = np.mean(real_features, axis=0)
        gen_mean = np.mean(gen_features, axis=0)
        
        real_cov = np.cov(real_features, rowvar=False)
        gen_cov = np.cov(gen_features, rowvar=False)
        
        # Compute FID
        diff = real_mean - gen_mean
        covmean = self._sqrtm(real_cov.dot(gen_cov))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(real_cov) + np.trace(gen_cov) - 2 * np.trace(covmean)
        return fid
    
    def _sqrtm(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        try:
            from scipy.linalg import sqrtm
            return sqrtm(matrix)
        except ImportError:
            # Fallback implementation
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            return eigenvecs.dot(np.diag(np.sqrt(eigenvals))).dot(eigenvecs.T)
    
    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images to [-1, 1] range."""
        if images.min() >= 0:
            return images * 2.0 - 1.0
        return images
    
    def _save_evaluation_samples(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        support_images: torch.Tensor,
        output_dir: str,
    ) -> None:
        """Save evaluation samples for visualization."""
        # Denormalize images
        real_images = (real_images + 1) / 2
        generated_images = (generated_images + 1) / 2
        support_images = (support_images + 1) / 2
        
        # Clamp to valid range
        real_images = torch.clamp(real_images, 0, 1)
        generated_images = torch.clamp(generated_images, 0, 1)
        support_images = torch.clamp(support_images, 0, 1)
        
        # Save sample grids
        self._save_image_grid(real_images[:16], os.path.join(output_dir, "real_samples.png"))
        self._save_image_grid(generated_images[:16], os.path.join(output_dir, "generated_samples.png"))
        self._save_image_grid(support_images[:16], os.path.join(output_dir, "support_samples.png"))
    
    def _save_image_grid(self, images: torch.Tensor, path: str) -> None:
        """Save a grid of images."""
        from torchvision.utils import make_grid
        
        grid = make_grid(images, nrow=4, normalize=False)
        grid_np = grid.permute(1, 2, 0).numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_metrics(self, metrics: Dict[str, float], output_dir: str) -> None:
        """Save metrics to file."""
        import json
        
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation metrics saved to {metrics_path}")
        print("Evaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    # Test the evaluator
    evaluator = FewShotEvaluator()
    
    # Create dummy data for testing
    real_images = torch.randn(100, 3, 64, 64)
    generated_images = torch.randn(100, 3, 64, 64)
    support_images = torch.randn(20, 3, 64, 64)
    
    # Test individual metrics
    fid_score = evaluator.compute_fid(real_images, generated_images)
    lpips_score = evaluator.compute_lpips(real_images, generated_images)
    pr_metrics = evaluator.compute_precision_recall(real_images, generated_images)
    fs_metrics = evaluator.compute_few_shot_metrics(support_images, real_images, generated_images)
    
    print(f"FID: {fid_score:.4f}")
    print(f"LPIPS: {lpips_score:.4f}")
    print(f"Precision/Recall: {pr_metrics}")
    print(f"Few-shot metrics: {fs_metrics}")
