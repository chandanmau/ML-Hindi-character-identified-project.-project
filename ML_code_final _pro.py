# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:30:00 2024

@author: satya
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score

# Configurations
train_dir = r"C:\Users\satya\OneDrive - Shiv Nadar Foundation\Desktop\dataset\train"
val_dir = r"C:\Users\satya\OneDrive - Shiv Nadar Foundation\Desktop\dataset\val"
label_file = r"C:\Users\satya\OneDrive - Shiv Nadar Foundation\Desktop\labs.txt"
batch_size = 32
img_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform_train = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_val = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load the training dataset
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_classes = len(train_dataset.classes)

# Custom dataset for validation
class CustomValDataset(Dataset):
    def __init__(self, val_dir, label_file, transform):
        self.image_paths = [os.path.join(val_dir, fname) for fname in os.listdir(val_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

        # Load labels from the label file
        self.labels_dict = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                img_name, true_label = line.strip().split()
                self.labels_dict[img_name] = true_label

        # Create class-to-index mapping based on training dataset
        self.class_to_idx = train_dataset.class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        img_name = os.path.basename(img_path)
        label = self.labels_dict.get(img_name, "Unknown")
        
        # Convert label to index
        label_idx = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)
        return image, label_idx

val_dataset = CustomValDataset(val_dir=val_dir, label_file=label_file, transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            flattened_size = self._get_flattened_size(dummy_input)

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flattened_size(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
model = CNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

# Validation loop
model.eval()
val_predictions = []
val_true_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        val_predictions.extend(preds.cpu().numpy())
        val_true_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(val_true_labels, val_predictions)
print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(val_true_labels, val_predictions, target_names=train_dataset.classes))

# Save the model
torch.save(model.state_dict(), 'hindi_letters_model.pth')
