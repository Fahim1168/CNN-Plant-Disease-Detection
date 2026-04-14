

# Import


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,Subset
from PIL import Image
import kagglehub

# Download Dataset

path = kagglehub.dataset_download("mohitsingh1804/plantvillage")
print("path",path)

#setup

torch.manual_seed(42)
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f" Using Device {device}")


TRAIN_PATH = os.path.join(path, "PlantVillage","train")
VAL_PATH = os.path.join(path, "PlantVillage","val")

# Transformation



transform = transforms.Compose(
    [
        transforms.Resize( (128,128) ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ]
)

# Custom Dataset

from genericpath import isfile

class MultiClassClassfication(Dataset):
  def __init__(self, root_dir,transform=None) :
    super().__init__()

    self.samples = []
    self.transform = transform

    self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])

    self.class_to_idx= {cls_name : idx  for idx,cls_name in enumerate(self.classes)}

    for class_name in self.classes:
      class_path = os.path.join(root_dir,class_name)

      for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path,img_name)

        if os.path.isfile(img_path):
          label = self.class_to_idx[class_name]
          self.samples.append( (img_path,label) )


  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx) :

    img_path, label = self.samples[idx]
    image = Image.open(img_path).convert("RGB")

    if self.transform:
      image = self.transform(image)


    return image, label

# example of enumerate

students = ['Alice', 'Bob', 'Charlie']
scores = [85, 92, 78]

# Using enumerate to get index and name
for idx, student in enumerate(students):
    print(f"{idx} → {student}, Score: {scores[idx]}")

# root/Apple___healthy/img1.jpg

# Example: {'Apple___healthy': 0, 'Tomato___late_blight': 1} → so label = 0 for 'Apple___healthy'


# [
#  ('root/Apple___healthy/img1.jpg', 0),
#  ('root/Apple___healthy/img2.jpg', 0),
#  ('root/Tomato___late_blight/img4.jpg', 1)
# ]



# <!-- Intuitive summary -->
# Outer loop → go through each class <br>
# Inner loop → go through each image in that class <br>
## Check → make sure it’s a file <br>
# Label → assign the numeric class <br>
# Store → save (image_path, label) for PyTorch <br>

# Load Data


# import os
# print(os.listdir("/kaggle/input/plantvillage"))

train_dataset_full = MultiClassClassfication(TRAIN_PATH, transform)
test_dataset_full = MultiClassClassfication(VAL_PATH, transform)
num_classes = len(train_dataset_full.classes)

print("Number of classes:", num_classes)
print("Full train size:", len(train_dataset_full))
print("Full test size:", len(test_dataset_full))
#
# Use a subset for fast testing (optional)
#


train_dataset = Subset(train_dataset_full, list(range(min(100, len(train_dataset_full)))))
test_dataset = Subset(test_dataset_full, list(range(min(100, len(test_dataset_full)))))


# DataLoader


pin = True if device.type == 'cuda' else False
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=pin)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=pin)

# CNN Architecture

class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MyCNN(num_classes=num_classes).to(device)


# Training Setup


learning_rate = 0.001
epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

