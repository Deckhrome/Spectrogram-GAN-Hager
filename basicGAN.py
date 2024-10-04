from torchviz import make_dot
import torch
from models import Discriminator2, Generator2

# Dimensions
z_dim = 1000

# Initialisation des modèles
G = Generator2(z_dim=z_dim, image_channels=3)
D = Discriminator2(image_channels=3)

# Génération d'une image factice à partir du générateur
z = torch.randn(1, z_dim)  # Vecteur latent de taille z_dim
fake_image = G(z)  # Générer une image à partir du vecteur latent

# Passer l'image générée dans le discriminateur
output = D(fake_image)

# Visualiser l'architecture du discriminateur
make_dot(output, params=dict(D.named_parameters())).render("discriminator_architecture", format="png")
