import torch
import os
from models import Generator2
from torchvision.utils import save_image

# Path to the generator model weights
generator_weights_pth = 'model_saved/generator_model.pth'

# Initialize the generator model
G = Generator2(z_dim=1000, image_channels=3)

# Load the saved weights securely using 'weights_only=True'
G.load_state_dict(torch.load(generator_weights_pth, map_location=torch.device('cpu'), weights_only=True))

# Generate an image using a random latent vector
latent_vector = torch.randn(1, 1000)  # Batch size = 1, z_dim = 1000
fake_img = G(latent_vector)

# Denormalize the generated image from [-1, 1] to [0, 1]
denormalized_img = (fake_img + 1) / 2

# Save the denormalized image
img_path = 'generated_image.png'
save_image(denormalized_img, img_path)

print(f"Image saved as {img_path}")
