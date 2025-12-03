# Few-Shot Image Generation

A production-ready implementation of few-shot image generation using meta-learning approaches. This project implements MAML (Model-Agnostic Meta-Learning) for rapid adaptation to new image domains with minimal examples.

## Features

- **Meta-Learning Framework**: Implements MAML for few-shot adaptation
- **Modern Architecture**: Spectral normalization, self-attention, and progressive growing
- **Comprehensive Evaluation**: FID, LPIPS, Precision/Recall, and few-shot specific metrics
- **Interactive Demo**: Streamlit-based web interface for easy experimentation
- **Production Ready**: Proper logging, checkpointing, and configuration management
- **Multiple Datasets**: Support for CIFAR-10, CelebA, and custom image folders

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Few-Shot-Image-Generation.git
cd Few-Shot-Image-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create toy dataset for testing:
```bash
python -c "from src.data.dataset import create_toy_dataset; create_toy_dataset('./data/toy_dataset')"
```

### Training

Train a model with default configuration:
```bash
python scripts/train.py --config configs/default.yaml --data_path ./data/toy_dataset
```

Train with advanced configuration:
```bash
python scripts/train.py --config configs/advanced.yaml --data_path ./data/cifar10
```

### Sampling

Generate samples from a trained model:
```bash
python scripts/sample.py --model_path ./checkpoints/few-shot-gen-last.ckpt --num_samples 16
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
0388_Few-shot_image_generation/
├── src/
│   ├── models/
│   │   └── few_shot_generator.py    # Model implementations
│   ├── data/
│   │   └── dataset.py               # Data loading utilities
│   └── utils/
│       └── evaluation.py            # Evaluation metrics
├── configs/
│   ├── default.yaml                 # Default configuration
│   └── advanced.yaml                # Advanced configuration
├── scripts/
│   ├── train.py                     # Training script
│   └── sample.py                    # Sampling script
├── demo/
│   └── app.py                       # Streamlit demo
├── tests/                           # Unit tests
├── assets/                          # Generated samples and visualizations
├── requirements.txt                  # Dependencies
└── README.md                        # This file
```

## Model Architecture

### FewShotGenerator
- Fully connected layers with spectral normalization
- Optional self-attention for improved quality
- Meta-learning adaptation capabilities
- Configurable architecture (layers, dimensions, etc.)

### FewShotDiscriminator
- Convolutional discriminator with spectral normalization
- Progressive growing for stable training
- Optional self-attention layers

### MetaLearner
- MAML-based meta-learning framework
- Rapid adaptation to new tasks
- Configurable adaptation steps and learning rates

## Configuration

The project uses YAML configuration files for easy experimentation:

```yaml
model:
  z_dim: 100                    # Latent dimension
  img_size: 64                  # Image size
  img_channels: 3               # Number of channels
  hidden_dim: 512               # Hidden layer dimension
  num_layers: 4                 # Number of layers
  use_spectral_norm: true       # Use spectral normalization
  use_self_attention: false     # Use self-attention
  meta_lr: 0.001                # Meta-learning rate
  adaptation_lr: 0.01           # Adaptation learning rate
  num_adaptation_steps: 5       # Number of adaptation steps

data:
  dataset_type: "cifar10"        # Dataset type
  batch_size: 4                 # Batch size
  num_shots: 5                  # Number of support samples
  num_query: 15                 # Number of query samples

training:
  max_epochs: 100                # Maximum training epochs
  checkpoint_dir: "./checkpoints" # Checkpoint directory
  early_stopping_patience: 20   # Early stopping patience
```

## Evaluation Metrics

The project provides comprehensive evaluation metrics:

### Standard Metrics
- **FID (Fréchet Inception Distance)**: Measures quality and diversity
- **LPIPS**: Perceptual similarity metric
- **Precision/Recall**: Generative model evaluation

### Few-Shot Specific Metrics
- **Adaptation Quality**: How well generated images match support images
- **Generation Quality**: How well generated images match query images
- **Diversity**: Diversity of generated samples

## Usage Examples

### Basic Generation
```python
from src.models.few_shot_generator import FewShotGenerator
import torch

# Create model
generator = FewShotGenerator(z_dim=100, img_size=64)

# Generate samples
z = torch.randn(16, 100)
samples = generator(z)
```

### Few-Shot Adaptation
```python
from src.models.few_shot_generator import MetaLearner

# Create meta-learner
meta_learner = MetaLearner(generator, discriminator)

# Adapt to new task
support_images = torch.randn(5, 3, 64, 64)  # 5 support images
adaptation_info = generator.adapt_to_task(support_images, num_adaptation_steps=5)

# Generate adapted samples
z = torch.randn(16, 100)
adapted_samples = generator(z)
```

### Evaluation
```python
from src.utils.evaluation import FewShotEvaluator

# Create evaluator
evaluator = FewShotEvaluator()

# Compute metrics
fid_score = evaluator.compute_fid(real_images, generated_images)
lpips_score = evaluator.compute_lpips(real_images, generated_images)
```

## Dataset Support

### CIFAR-10
```bash
python scripts/train.py --config configs/default.yaml --data_path ./data/cifar10
```

### CelebA
```bash
python scripts/train.py --config configs/advanced.yaml --data_path ./data/celeba
```

### Custom Dataset
Place your images in a folder structure:
```
data/
└── custom_dataset/
    ├── class_0/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── class_1/
        ├── image1.jpg
        └── image2.jpg
```

Then train with:
```bash
python scripts/train.py --config configs/default.yaml --data_path ./data/custom_dataset
```

## Advanced Features

### Self-Attention
Enable self-attention for improved quality:
```yaml
model:
  use_self_attention: true
```

### Mixed Precision Training
The training script automatically uses mixed precision when available:
```python
trainer = pl.Trainer(precision=16)  # Mixed precision
```

### Logging and Monitoring
- **TensorBoard**: Built-in TensorBoard logging
- **Weights & Biases**: Optional W&B integration
- **Checkpointing**: Automatic model checkpointing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use gradient accumulation
   - Enable mixed precision training

2. **Training Instability**
   - Enable spectral normalization
   - Reduce learning rates
   - Increase gradient clipping

3. **Poor Generation Quality**
   - Increase model capacity
   - Enable self-attention
   - Adjust loss weights

### Performance Tips

1. **Faster Training**
   - Use multiple GPUs with DDP
   - Enable mixed precision
   - Use gradient accumulation

2. **Better Quality**
   - Increase model size
   - Enable self-attention
   - Use more adaptation steps

3. **Memory Efficiency**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{few_shot_image_generation,
  title={Few-Shot Image Generation with Meta-Learning},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Few-Shot-Image-Generation}
}
```

## Acknowledgments

- PyTorch Lightning for the training framework
- Clean-FID for FID computation
- LPIPS for perceptual similarity
- Streamlit for the interactive demo

## Model Card

### Intended Use
This model is designed for research and educational purposes in few-shot image generation. It can be used to:
- Generate images in new domains with minimal examples
- Study meta-learning approaches for generative models
- Create interactive demos for image generation

### Limitations
- Generated images may not be photorealistic
- Performance depends on the quality and diversity of support images
- May require fine-tuning for specific domains

### Bias and Fairness
- The model may inherit biases from training data
- Generated images should be evaluated for appropriateness
- Consider using diverse support images for adaptation

### Safety Considerations
- Generated content should be reviewed before publication
- Consider watermarking generated images
- Be aware of potential misuse for creating misleading content
# Few-Shot-Image-Generation
