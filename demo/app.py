"""
Streamlit demo for few-shot image generation.

This demo provides an interactive interface for generating images
with few-shot adaptation capabilities.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tempfile
from typing import Optional, List
import yaml
from omegaconf import OmegaConf

from src.models.few_shot_generator import FewShotGenerator, FewShotDiscriminator, MetaLearner
from src.data.dataset import FewShotDataLoader, create_toy_dataset
from src.utils.evaluation import FewShotEvaluator


@st.cache_resource
def load_model(model_path: str, config_path: str):
    """Load the trained model with caching."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    # Load configuration
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        config = OmegaConf.create({
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
    
    # Create and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = FewShotGenerator(**config.model)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            generator.load_state_dict(checkpoint)
        
        generator.to(device)
        generator.eval()
        
        return generator, config, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def generate_samples(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    num_samples: int = 16,
    seed: int = 42,
    temperature: float = 1.0,
    truncation: float = 1.0,
) -> torch.Tensor:
    """Generate samples from the model."""
    torch.manual_seed(seed)
    
    with torch.no_grad():
        # Sample noise
        z = torch.randn(num_samples, config.model.z_dim, device=device)
        
        # Apply truncation
        if truncation < 1.0:
            z = torch.clamp(z, -truncation, truncation)
        
        # Apply temperature
        if temperature != 1.0:
            z = z * temperature
        
        # Generate images
        generated_images = model(z)
        
        # Denormalize to [0, 1]
        generated_images = (generated_images + 1) / 2
        generated_images = torch.clamp(generated_images, 0, 1)
    
    return generated_images


def generate_with_adaptation(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    support_images: torch.Tensor,
    num_samples: int = 16,
    num_adaptation_steps: int = 5,
    adaptation_lr: float = 0.01,
    seed: int = 42,
) -> torch.Tensor:
    """Generate samples with few-shot adaptation."""
    torch.manual_seed(seed)
    
    # Store original parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Adapt model to support images
    adaptation_info = model.adapt_to_task(
        support_images.to(device),
        num_adaptation_steps=num_adaptation_steps,
        adaptation_lr=adaptation_lr,
    )
    
    # Generate samples
    generated_images = generate_samples(model, config, device, num_samples, seed)
    
    # Restore original parameters
    for name, param in model.named_parameters():
        param.data = original_params[name].data
    
    return generated_images, adaptation_info


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Convert from (C, H, W) to (H, W, C)
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    # Convert to uint8
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def create_image_grid(images: torch.Tensor, nrow: int = 4) -> Image.Image:
    """Create a grid of images."""
    from torchvision.utils import make_grid
    
    grid = make_grid(images, nrow=nrow, normalize=False)
    return tensor_to_pil(grid)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Few-Shot Image Generation",
        page_icon="🎨",
        layout="wide",
    )
    
    st.title("🎨 Few-Shot Image Generation Demo")
    st.markdown("Generate images with few-shot adaptation capabilities using meta-learning.")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="./checkpoints/few-shot-gen-last.ckpt",
        help="Path to the trained model checkpoint"
    )
    
    config_path = st.sidebar.text_input(
        "Config Path",
        value="./configs/default.yaml",
        help="Path to the configuration file"
    )
    
    # Load model
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            model_data = load_model(model_path, config_path)
            if model_data is not None:
                st.session_state.model, st.session_state.config, st.session_state.device = model_data
                st.sidebar.success("Model loaded successfully!")
            else:
                st.sidebar.error("Failed to load model!")
    
    # Check if model is loaded
    if 'model' not in st.session_state:
        st.warning("Please load a model first using the sidebar.")
        st.stop()
    
    model = st.session_state.model
    config = st.session_state.config
    device = st.session_state.device
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Basic Generation", "Few-Shot Adaptation", "Model Info"])
    
    with tab1:
        st.header("Basic Image Generation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Generation Parameters")
            
            num_samples = st.slider("Number of Samples", 1, 32, 16)
            seed = st.number_input("Random Seed", value=42, min_value=0, max_value=1000)
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
            truncation = st.slider("Truncation", 0.1, 2.0, 1.0, 0.1)
            
            if st.button("Generate Images"):
                with st.spinner("Generating images..."):
                    samples = generate_samples(
                        model, config, device, num_samples, seed, temperature, truncation
                    )
                    st.session_state.basic_samples = samples
        
        with col2:
            st.subheader("Generated Images")
            
            if 'basic_samples' in st.session_state:
                samples = st.session_state.basic_samples
                
                # Create grid
                grid_img = create_image_grid(samples, nrow=int(np.sqrt(len(samples))))
                
                st.image(grid_img, caption="Generated Images", use_column_width=True)
                
                # Download button
                img_buffer = io.BytesIO()
                grid_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="Download Generated Images",
                    data=img_buffer.getvalue(),
                    file_name="generated_images.png",
                    mime="image/png"
                )
            else:
                st.info("Click 'Generate Images' to create samples.")
    
    with tab2:
        st.header("Few-Shot Adaptation")
        st.markdown("Upload a few example images to adapt the model to your specific style.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Support Images")
            
            uploaded_files = st.file_uploader(
                "Upload support images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload 3-10 images that represent the style you want to generate"
            )
            
            if uploaded_files:
                support_images = []
                for uploaded_file in uploaded_files:
                    img = Image.open(uploaded_file).convert('RGB')
                    # Resize to model input size
                    img = img.resize((config.model.img_size, config.model.img_size))
                    # Convert to tensor
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
                    support_images.append(img_tensor)
                
                support_tensor = torch.stack(support_images)
                
                # Display support images
                st.write("Support Images:")
                for i, img in enumerate(support_images):
                    pil_img = tensor_to_pil((img + 1) / 2)
                    st.image(pil_img, caption=f"Support Image {i+1}", width=100)
                
                st.session_state.support_images = support_tensor
        
        with col2:
            st.subheader("Adaptation Parameters")
            
            num_samples = st.slider("Number of Generated Samples", 1, 32, 16, key="adapt_samples")
            num_adaptation_steps = st.slider("Adaptation Steps", 1, 20, 5, key="adapt_steps")
            adaptation_lr = st.slider("Adaptation Learning Rate", 0.001, 0.1, 0.01, 0.001, key="adapt_lr")
            seed = st.number_input("Random Seed", value=42, min_value=0, max_value=1000, key="adapt_seed")
            
            if st.button("Generate with Adaptation") and 'support_images' in st.session_state:
                with st.spinner("Adapting model and generating images..."):
                    generated_images, adaptation_info = generate_with_adaptation(
                        model, config, device,
                        st.session_state.support_images,
                        num_samples, num_adaptation_steps, adaptation_lr, seed
                    )
                    st.session_state.adapted_samples = generated_images
                    st.session_state.adaptation_info = adaptation_info
        
        # Display results
        if 'adapted_samples' in st.session_state:
            st.subheader("Adapted Generation Results")
            
            samples = st.session_state.adapted_samples
            grid_img = create_image_grid(samples, nrow=int(np.sqrt(len(samples))))
            
            st.image(grid_img, caption="Adapted Generated Images", use_column_width=True)
            
            # Show adaptation info
            if 'adaptation_info' in st.session_state:
                info = st.session_state.adaptation_info
                st.write("Adaptation Information:")
                st.write(f"- Final Loss: {info['final_loss']:.4f}")
                st.write(f"- Adaptation Steps: {len(info['adaptation_losses'])}")
                
                # Plot adaptation loss curve
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(info['adaptation_losses'])
                ax.set_xlabel('Adaptation Step')
                ax.set_ylabel('Loss')
                ax.set_title('Adaptation Loss Curve')
                st.pyplot(fig)
            
            # Download button
            img_buffer = io.BytesIO()
            grid_img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="Download Adapted Images",
                data=img_buffer.getvalue(),
                file_name="adapted_images.png",
                mime="image/png"
            )
    
    with tab3:
        st.header("Model Information")
        
        st.subheader("Model Architecture")
        st.write(f"- **Latent Dimension**: {config.model.z_dim}")
        st.write(f"- **Image Size**: {config.model.img_size}x{config.model.img_size}")
        st.write(f"- **Image Channels**: {config.model.img_channels}")
        st.write(f"- **Hidden Dimension**: {config.model.hidden_dim}")
        st.write(f"- **Number of Layers**: {config.model.num_layers}")
        st.write(f"- **Spectral Normalization**: {config.model.use_spectral_norm}")
        st.write(f"- **Self-Attention**: {config.model.use_self_attention}")
        
        st.subheader("Training Configuration")
        st.write(f"- **Meta Learning Rate**: {config.model.meta_lr}")
        st.write(f"- **Adaptation Learning Rate**: {config.model.adaptation_lr}")
        st.write(f"- **Adaptation Steps**: {config.model.num_adaptation_steps}")
        st.write(f"- **Reconstruction Weight**: {config.model.lambda_recon}")
        st.write(f"- **Perceptual Weight**: {config.model.lambda_perceptual}")
        
        st.subheader("Device Information")
        st.write(f"- **Device**: {device}")
        st.write(f"- **CUDA Available**: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"- **CUDA Device**: {torch.cuda.get_device_name()}")
            st.write(f"- **CUDA Memory**: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
