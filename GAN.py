import os
import time  
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset  
from torchvision.utils import save_image
from models import Generator2, Discriminator2

# Création des répertoires s'ils n'existent pas
os.makedirs('model_saved', exist_ok=True)
os.makedirs('samples', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformation des images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Chargement des images
image_dir = 'sampleSpectrograms'
full_dataset = ImageFolder(root=image_dir, transform=transform)

# Limiter le dataset à max_images
max_images = 50000
dataset_size = len(full_dataset)
indices = list(range(min(max_images, dataset_size)))  
train_dataset = Subset(full_dataset, indices)

# DataLoader pour charger les images
batch_size = 16
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Paramètres du GAN
z_dim = 1000
image_dim = (3, 266, 126)

# Initialisation du générateur et du discriminateur
G = Generator2(z_dim=z_dim, image_channels=image_dim[0]).to(device)
D = Discriminator2(image_channels=image_dim[0]).to(device)

# Optimiseurs
lr_G = 0.0002
lr_D = 0.0001
G_optimizer = optim.Adam(G.parameters(), lr=lr_G)
D_optimizer = optim.Adam(D.parameters(), lr=lr_D)

# Fonction de perte
criterion = nn.BCELoss()

# Fonction d'entraînement du Discriminateur
def D_train(x):
    D.zero_grad()
    x_real = x.to(device)
    batch_size = x.size(0)
    y_real = torch.ones(batch_size, 1).to(device)

    D_output_real = D(x_real)
    D_real_loss = criterion(D_output_real, y_real)

    z = torch.randn(batch_size, z_dim).to(device)
    x_fake = G(z)
    y_fake = torch.zeros(batch_size, 1).to(device)

    D_output_fake = D(x_fake.detach())
    D_fake_loss = criterion(D_output_fake, y_fake)

    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.item()

# Fonction d'entraînement du Générateur
def G_train(x):
    G.zero_grad()
    batch_size = x.size(0)

    z = torch.randn(batch_size, z_dim).to(device)
    G_output = G(z)
    D_output = D(G_output)
    y = torch.ones(batch_size, 1).to(device)
    G_loss = criterion(D_output, y)
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

# Entraînement du GAN
num_epochs = 100
start_time = time.time()  # Mesurer le temps total

D_losses, G_losses = [], []  # Pour enregistrer toutes les pertes

# Start training
print('Starting Training Loop...')

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()  # Mesurer le temps d'une époque
    D_losses_epoch, G_losses_epoch = [], []

    for batch_idx, (x, _) in enumerate(train_loader):
        D_losses_epoch.append(D_train(x))
        G_losses_epoch.append(G_train(x))

    D_losses.append(torch.mean(torch.FloatTensor(D_losses_epoch)))
    G_losses.append(torch.mean(torch.FloatTensor(G_losses_epoch)))

    # Sauvegarder les images générées tous les 10 epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            z = torch.randn(1, z_dim).to(device)
            fake_images = G(z)
            save_image(fake_images, f'samples/sample_epoch_{epoch}.png', nrow=4, normalize=True)

    # Calcul du temps pour l'époque
    epoch_time = time.time() - epoch_start_time
    print(f'Epoch [{epoch}/{num_epochs}]: loss_d: {D_losses[-1]:.3f}, loss_g: {G_losses[-1]:.3f}, epoch_time: {epoch_time:.2f}s')

# Calcul du temps total d'entraînement
total_time = time.time() - start_time
print(f'Total training time: {total_time / 60:.2f} minutes')

# Sauvegarde des modèles entraînés
torch.save(G.state_dict(), 'model_saved/generator_model.pth')
torch.save(D.state_dict(), 'model_saved/discriminator_model.pth')

# Traçage des pertes
plt.plot(D_losses, label='Discriminator Loss')
plt.plot(G_losses, label='Generator Loss')
plt.legend()
plt.savefig('losses.png')
plt.show()