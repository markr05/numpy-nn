import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchmetrics.classification import Accuracy, Precision, Recall

import random

batch_size = 60
num_classes = 15
transform = transforms.Compose([
    # transforms.RandomRotation(degrees=(10, 15)),  # Rotate between 10° and 15°
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_dataset = ImageFolder(root="mnist_images/", transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImageFolder(root="mnist_images/", transform=transform)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(CNN, self).__init__()
    # first convolutional layer
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
    # first maxpooling layer
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    # second convolutional layer
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    # linear layer (like in rnn)
    self.fc1 = nn.Linear(16*7*7, num_classes)

  def forward(self, x): # x is the input tensor
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)
    return x
  
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(in_channels=1, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4

for epoch in range(num_epochs):
  print(f"Epoch [{epoch+1}/{num_epochs}]")
  
  correct = 0
  total = 0
  
  for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
    data = data.to(device)
    targets = targets.to(device)
    scores = model(data)
    loss = criterion(scores, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Compute batch accuracy
    _, predicted = torch.max(scores, 1)
    correct += (predicted == targets).sum().item()
    total += targets.size(0)
  
  # Epoch accuracy
  epoch_accuracy = correct / total
  print(f"Epoch [{epoch+1}/{num_epochs}] Accuracy: {epoch_accuracy:.4f}")

acc = Accuracy(task="multiclass", num_classes=num_classes)
precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

model.eval()
with torch.no_grad():
  for images, labels in test_loader:
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    acc(preds, labels)
    precision(preds, labels)
    recall(preds, labels)

test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")

torch.save(model.state_dict(), 'MATHCNN5_less_epochs.pth')