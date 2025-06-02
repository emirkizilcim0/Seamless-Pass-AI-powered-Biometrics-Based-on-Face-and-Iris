import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# Dataset class
class LandmarkDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_path = self.landmarks_frame.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")

        landmarks = self.landmarks_frame.iloc[idx, 1:].values.astype('float32')
        landmarks = torch.tensor(landmarks).view(-1, 2)  # (68, 2)

        if self.transform:
            image = self.transform(image)

        return image, landmarks

# Model definition: ResNet18 with modified fc layer
def get_resnet18_model(num_landmarks=68, pretrained=False):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_landmarks * 2)  # 136 outputs
    return model

# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, landmarks in train_loader:
            images = images.to(device)
            landmarks = landmarks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.view(-1, 68, 2)  # reshape to (batch, 68, 2)

            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

    print("Training complete!")

# Parameters
batch_size = 16
num_epochs = 25
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (resize + normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = LandmarkDataset('training_landmarks_norm.csv', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Instantiate model
model = get_resnet18_model(pretrained=False).to(device) # No pretrained weights

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

# Optionally save the trained model
torch.save(model.state_dict(), "resnet18_landmark.pth")
