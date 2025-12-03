"""
Unit tests for few-shot image generation models.

This module provides comprehensive tests for all components of the
few-shot image generation system.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from src.models.few_shot_generator import (
    FewShotGenerator, 
    FewShotDiscriminator, 
    MetaLearner,
    SelfAttention
)
from src.data.dataset import FewShotDataset, FewShotDataLoader, create_toy_dataset
from src.utils.evaluation import FewShotEvaluator, InceptionFeatureExtractor


class TestFewShotGenerator:
    """Test cases for FewShotGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization with different configurations."""
        # Basic configuration
        generator = FewShotGenerator(z_dim=100, img_size=64)
        assert generator.z_dim == 100
        assert generator.img_size == 64
        assert generator.img_channels == 3
        
        # Advanced configuration
        generator = FewShotGenerator(
            z_dim=128, 
            img_size=128, 
            img_channels=1,
            hidden_dim=1024,
            num_layers=6,
            use_spectral_norm=True,
            use_self_attention=True
        )
        assert generator.z_dim == 128
        assert generator.img_size == 128
        assert generator.img_channels == 1
        assert generator.use_self_attention == True
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = FewShotGenerator(z_dim=100, img_size=64)
        
        # Test forward pass
        batch_size = 4
        z = torch.randn(batch_size, 100)
        output = generator(z)
        
        assert output.shape == (batch_size, 3, 64, 64)
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_generator_adaptation(self):
        """Test generator adaptation to task."""
        generator = FewShotGenerator(z_dim=100, img_size=64)
        
        # Create support images
        support_images = torch.randn(5, 3, 64, 64)
        
        # Test adaptation
        adaptation_info = generator.adapt_to_task(
            support_images, 
            num_adaptation_steps=3,
            adaptation_lr=0.01
        )
        
        assert 'original_params' in adaptation_info
        assert 'adaptation_losses' in adaptation_info
        assert 'final_loss' in adaptation_info
        assert len(adaptation_info['adaptation_losses']) == 3
        assert isinstance(adaptation_info['final_loss'], float)


class TestFewShotDiscriminator:
    """Test cases for FewShotDiscriminator."""
    
    def test_discriminator_initialization(self):
        """Test discriminator initialization."""
        discriminator = FewShotDiscriminator(
            img_channels=3,
            img_size=64,
            hidden_dim=512,
            num_layers=4,
            use_spectral_norm=True
        )
        
        assert discriminator.img_channels == 3
        assert discriminator.img_size == 64
        assert discriminator.hidden_dim == 512
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = FewShotDiscriminator(img_channels=3, img_size=64)
        
        # Test forward pass
        batch_size = 4
        images = torch.randn(batch_size, 3, 64, 64)
        output = discriminator(images)
        
        assert output.shape == (batch_size, 1)


class TestSelfAttention:
    """Test cases for SelfAttention module."""
    
    def test_self_attention_initialization(self):
        """Test self-attention initialization."""
        attention = SelfAttention(channels=64)
        assert attention.channels == 64
        assert attention.m == 8  # channels // 8
    
    def test_self_attention_forward(self):
        """Test self-attention forward pass."""
        attention = SelfAttention(channels=64)
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 64, 32, 32)
        output = attention(x)
        
        assert output.shape == x.shape
        assert torch.allclose(output, x, atol=1e-6)  # Should be close to input initially


class TestMetaLearner:
    """Test cases for MetaLearner."""
    
    def test_meta_learner_initialization(self):
        """Test meta-learner initialization."""
        generator = FewShotGenerator(z_dim=100, img_size=64)
        discriminator = FewShotDiscriminator(img_channels=3, img_size=64)
        
        meta_learner = MetaLearner(
            generator=generator,
            discriminator=discriminator,
            meta_lr=0.001,
            adaptation_lr=0.01
        )
        
        assert meta_learner.meta_lr == 0.001
        assert meta_learner.adaptation_lr == 0.01
    
    def test_meta_train_step(self):
        """Test meta-training step."""
        generator = FewShotGenerator(z_dim=100, img_size=64)
        discriminator = FewShotDiscriminator(img_channels=3, img_size=64)
        
        meta_learner = MetaLearner(generator, discriminator)
        
        # Create mock tasks
        support_tasks = [torch.randn(4, 3, 64, 64)]
        query_tasks = [torch.randn(4, 3, 64, 64)]
        
        # Test meta-training step
        losses = meta_learner.meta_train_step(
            support_tasks=support_tasks,
            query_tasks=query_tasks,
            num_adaptation_steps=3
        )
        
        assert 'meta_loss' in losses
        assert 'avg_task_loss' in losses
        assert 'task_losses' in losses
        assert isinstance(losses['meta_loss'], float)
    
    def test_generate_samples(self):
        """Test sample generation."""
        generator = FewShotGenerator(z_dim=100, img_size=64)
        discriminator = FewShotDiscriminator(img_channels=3, img_size=64)
        
        meta_learner = MetaLearner(generator, discriminator)
        
        # Test sample generation
        samples = meta_learner.generate_samples(num_samples=8)
        
        assert samples.shape == (8, 3, 64, 64)


class TestFewShotDataset:
    """Test cases for FewShotDataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        # This test requires a mock dataset or toy dataset
        with patch('src.data.dataset.datasets.CIFAR10') as mock_cifar10:
            mock_cifar10.return_value = [(torch.randn(3, 32, 32), 0) for _ in range(100)]
            
            dataset = FewShotDataset(
                data_path='./test_data',
                dataset_type='cifar10',
                num_shots=5,
                num_query=15
            )
            
            assert dataset.num_shots == 5
            assert dataset.num_query == 15
            assert dataset.dataset_type == 'cifar10'
    
    def test_create_toy_dataset(self):
        """Test toy dataset creation."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            create_toy_dataset(
                output_path=temp_dir,
                num_classes=3,
                samples_per_class=10,
                img_size=32
            )
            
            # Check if directories were created
            for i in range(3):
                class_dir = os.path.join(temp_dir, f"class_{i}")
                assert os.path.exists(class_dir)
                
                # Check if images were created
                images = [f for f in os.listdir(class_dir) if f.endswith('.png')]
                assert len(images) == 10


class TestFewShotEvaluator:
    """Test cases for FewShotEvaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = FewShotEvaluator()
        assert evaluator.device is not None
        assert evaluator.inception_extractor is not None
        assert evaluator.lpips_model is not None
    
    def test_fid_computation(self):
        """Test FID computation."""
        evaluator = FewShotEvaluator()
        
        # Create dummy data
        real_images = torch.randn(50, 3, 64, 64)
        generated_images = torch.randn(50, 3, 64, 64)
        
        # Test FID computation
        fid_score = evaluator.compute_fid(real_images, generated_images)
        
        assert isinstance(fid_score, float)
        assert fid_score >= 0
    
    def test_lpips_computation(self):
        """Test LPIPS computation."""
        evaluator = FewShotEvaluator()
        
        # Create dummy data
        real_images = torch.randn(10, 3, 64, 64)
        generated_images = torch.randn(10, 3, 64, 64)
        
        # Test LPIPS computation
        lpips_score = evaluator.compute_lpips(real_images, generated_images)
        
        assert isinstance(lpips_score, float)
        assert lpips_score >= 0
    
    def test_precision_recall_computation(self):
        """Test precision/recall computation."""
        evaluator = FewShotEvaluator()
        
        # Create dummy data
        real_images = torch.randn(20, 3, 64, 64)
        generated_images = torch.randn(20, 3, 64, 64)
        
        # Test precision/recall computation
        pr_metrics = evaluator.compute_precision_recall(real_images, generated_images)
        
        assert 'precision' in pr_metrics
        assert 'recall' in pr_metrics
        assert 'f1' in pr_metrics
        assert 0 <= pr_metrics['precision'] <= 1
        assert 0 <= pr_metrics['recall'] <= 1
    
    def test_few_shot_metrics(self):
        """Test few-shot specific metrics."""
        evaluator = FewShotEvaluator()
        
        # Create dummy data
        support_images = torch.randn(5, 3, 64, 64)
        query_images = torch.randn(10, 3, 64, 64)
        generated_images = torch.randn(10, 3, 64, 64)
        
        # Test few-shot metrics
        fs_metrics = evaluator.compute_few_shot_metrics(
            support_images, query_images, generated_images
        )
        
        assert 'adaptation_quality' in fs_metrics
        assert 'generation_quality' in fs_metrics
        assert 'diversity' in fs_metrics
        assert 'adaptation_generation_ratio' in fs_metrics


class TestInceptionFeatureExtractor:
    """Test cases for InceptionFeatureExtractor."""
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = InceptionFeatureExtractor()
        assert extractor.device is not None
        assert extractor.inception is not None
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        extractor = InceptionFeatureExtractor()
        
        # Create dummy images
        images = torch.randn(4, 3, 64, 64)
        
        # Test feature extraction
        features = extractor(images)
        
        assert features.shape == (4, 2048)  # Inception v3 feature dimension


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_training_pipeline(self):
        """Test complete training pipeline."""
        # This is a simplified integration test
        generator = FewShotGenerator(z_dim=100, img_size=64)
        discriminator = FewShotDiscriminator(img_channels=3, img_size=64)
        meta_learner = MetaLearner(generator, discriminator)
        
        # Create dummy data
        support_images = torch.randn(4, 3, 64, 64)
        query_images = torch.randn(4, 3, 64, 64)
        
        # Test training step
        losses = meta_learner.meta_train_step(
            support_tasks=[support_images],
            query_tasks=[query_images],
            num_adaptation_steps=2
        )
        
        assert 'meta_loss' in losses
        assert isinstance(losses['meta_loss'], float)
    
    def test_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        evaluator = FewShotEvaluator()
        
        # Create dummy data
        real_images = torch.randn(20, 3, 64, 64)
        generated_images = torch.randn(20, 3, 64, 64)
        support_images = torch.randn(5, 3, 64, 64)
        
        # Test all metrics
        fid_score = evaluator.compute_fid(real_images, generated_images)
        lpips_score = evaluator.compute_lpips(real_images, generated_images)
        pr_metrics = evaluator.compute_precision_recall(real_images, generated_images)
        fs_metrics = evaluator.compute_few_shot_metrics(
            support_images, real_images, generated_images
        )
        
        # All metrics should be computable
        assert isinstance(fid_score, float)
        assert isinstance(lpips_score, float)
        assert isinstance(pr_metrics, dict)
        assert isinstance(fs_metrics, dict)


# Utility functions for testing
def create_mock_dataset(num_samples=100):
    """Create a mock dataset for testing."""
    images = [torch.randn(3, 64, 64) for _ in range(num_samples)]
    labels = [i % 10 for i in range(num_samples)]
    return images, labels


def create_mock_batch(batch_size=4, num_shots=5, num_query=15):
    """Create a mock batch for testing."""
    return {
        'support_images': torch.randn(batch_size, num_shots, 3, 64, 64),
        'query_images': torch.randn(batch_size, num_query, 3, 64, 64),
        'class_label': torch.randint(0, 10, (batch_size,)),
        'task_type': ['generation'] * batch_size
    }


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
