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
        self.fc = nn.Linear(z_dim, hidden_dim * 8 * 11 * 16)  # Adjusting the dimension to start from a smaller feature map

        # Convolutional Transpose Layers
        self.ct1 = nn.ConvTranspose2d(hidden_dim * 16, hidden_dim * 8, kernel_size=(4,3), stride=(2,1), padding=1)  # 8x11 -> 16x11
        self.bn1 = nn.BatchNorm2d(hidden_dim * 8)
        
        self.ct2 = nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=(4,3), stride=(2,1), padding=1)  # 16x11 -> 32x11
        self.bn2 = nn.BatchNorm2d(hidden_dim * 4)
        
        self.ct3 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=(4,3), stride=(2, 1), padding=1)  # 32x11 -> 64x11
        self.bn3 = nn.BatchNorm2d(hidden_dim * 2)
        
        self.ct4 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=(4,3), stride=(2, 1), padding= 1)  # 64x11 -> 128x11
        self.bn4 = nn.BatchNorm2d(hidden_dim)

        self.ct5 = nn.ConvTranspose2d(hidden_dim, image_channels, kernel_size= 3, stride=1, padding=1)  # Final to 126x266
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 16 * 8, 8, 11)  # Start with a 8x8 feature map
        x = F.relu(self.bn1(self.ct1(x)))  # 16x16
        x = F.relu(self.bn2(self.ct2(x)))  # 32x32
        x = F.relu(self.bn3(self.ct3(x)))
        x = F.relu(self.bn4(self.ct4(x)))
        x = torch.tanh(self.ct5(x))        # Output size 126x266
        return x


class Discriminator2(nn.Module):
    def __init__(self, image_channels, hidden_dim=64):
        super(Discriminator2, self).__init__()
        self.image_channels = image_channels
        
        # Adjusted convolutional layers to handle input of size (2, 128, 11)
        self.conv1 = nn.Conv2d(image_channels, hidden_dim, kernel_size=3, stride=2, padding=1)  # Input: (image_channels, 128, 11) -> Output: (hidden_dim, 64, 6)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)  # Output: (hidden_dim*2, 32, 3)
        self.conv2_drop = nn.Dropout2d()
        
        # Adjust fc_input_dim to match the new dimensions after convolutions
        self.fc_input_dim = hidden_dim * 2 * 32 * 3  # (32, 3) after two convolutions

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Reshape input to (batch_size, image_channels, 128, 11)
        x = x.view(-1, self.image_channels, 128, 11)
        x = F.relu(self.conv1(x))  # Output: (hidden_dim, 64, 6)
        x = F.relu(self.conv2_drop(self.conv2(x)))  # Output: (hidden_dim*2, 32, 3)
        
        # Reshape the output to flatten for the fully connected layers
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return torch.sigmoid(x)
