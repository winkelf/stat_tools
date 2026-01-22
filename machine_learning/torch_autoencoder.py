import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)  # Bottleneck layer
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()  # Output pixel values between 0 and 1
        )

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)  # Reshape to original image size

# Initialize the model, loss function, and optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data  # We only need images, not labels
        img = img.to(device)

        # Forward pass
        output = model(img)
        loss = criterion(output, img)  # Reconstruction loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

# Sample visualization
import matplotlib.pyplot as plt

# Function to display original and reconstructed images
def visualize_reconstruction(model, data_loader):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            img, _ = batch
            img = img.to(device)
            output = model(img)
            
            # Display original images
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(img[0].cpu().squeeze(), cmap='gray')
            axes[0].set_title("Original Image")

            # Display reconstructed images
            axes[1].imshow(output[0].cpu().squeeze(), cmap='gray')
            axes[1].set_title("Reconstructed Image")

            plt.show()
            break

# Visualize some reconstructions
visualize_reconstruction(model, train_loader)

