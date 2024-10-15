import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Generator2(nn.Module):
    def __init__(self, z_dim, image_channels, hidden_dim=8):
        super(Generator2, self).__init__()
        self.z_dim = z_dim
        self.image_channels = image_channels
        
        # Dimension after the linear layer
        self.fc = nn.Linear(z_dim, hidden_dim * 8 * 8 * 16)  # Adjusting the dimension to start from a smaller feature map

        # Convolutional Transpose Layers
        self.ct1 = nn.ConvTranspose2d(hidden_dim * 16, hidden_dim * 8, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.bn1 = nn.BatchNorm2d(hidden_dim * 8)
        
        self.ct2 = nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        self.bn2 = nn.BatchNorm2d(hidden_dim * 4)
        
        self.ct3 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=(2, 2), padding= (0,1))  # 32x32 -> 64x66
        
        self.ct4 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=(2, 2), padding= (0,1))  # 64x64 -> 128x134
        
        self.ct5 = nn.ConvTranspose2d(hidden_dim, image_channels, kernel_size=(2, 3), stride=(2, 1), padding=(1, 2))  # Final to 126x266
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16 * 8, 8, 8)  # Start with a 8x8 feature map
        x = F.relu(self.bn1(self.ct1(x)))  # 16x16
        x = F.relu(self.bn2(self.ct2(x)))  # 32x32
        x = F.relu(self.ct3(x))            # 64x64
        x = F.relu(self.ct4(x))            # 128x128
        x = torch.tanh(self.ct5(x))        # Output size 126x266
        
        return x




class Discriminator2(nn.Module):
    def __init__(self, image_channels, hidden_dim=64):
        super(Discriminator2, self).__init__()
        self.image_channels = image_channels
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(image_channels, hidden_dim, kernel_size=3, stride=2, padding=1)  # Output: (hidden_dim, 126/2, 266/2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)  # Output: (hidden_dim*2, 126/4, 266/4)
        self.conv2_drop = nn.Dropout2d()
        
        # Calculate the input dimension for the fully connected layer
        self.fc_input_dim = hidden_dim * 2 * 32 * 67  # 126/4 et 266/4 après 2 convolutions avec stride=2

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Apply convolutional layers with max pooling
        x = F.relu(self.conv1(x)) # 63x133
        x = F.relu(self.conv2_drop(self.conv2(x)))
        
        # Reshape output for fully connected layer
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the tensor

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return torch.sigmoid(x)