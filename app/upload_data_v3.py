import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import tarfile
from io import BytesIO
from aistore.client import Client

number_of_classes = 10

bucket_name = "imagenet-data"
tar_file_path ="train_images_0.tar.gz"
ais_client = Client("http://localhost:51080")


class AISTarDataset(Dataset):
    def __init__(self, client, bucket_name, tar_path, transform=None):
        self.client = client
        self.bucket_name = bucket_name
        self.tar_path = tar_path
        self.transform = transform
        self.members = []

        # Download the tar file from AIStore
        self.download_tar()

        # Open the tar file and get the member list
        with tarfile.open(fileobj=BytesIO(self.tar_content), mode='r:gz') as tar:
            self.members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.jpg')]

    def download_tar(self):
        obj = self.client.bucket(self.bucket_name).object(self.tar_path)
        response = obj.get()
        self.tar_content = response.read()

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        member = self.members[idx]
        with tarfile.open(fileobj=BytesIO(self.tar_content), mode='r:gz') as tar:
            img_file = tar.extractfile(member)
            img = Image.open(BytesIO(img_file.read()))
            if self.transform:
                img = self.transform(img)
        return img


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


dataset = AISTarDataset(ais_client, bucket_name, tar_file_path, transform=transform)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, number_of_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


def train_model(dataloader, model, criterion, optimizer, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.zeros(len(inputs), dtype=torch.long))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss {running_loss/len(dataloader)}')


train_model(train_loader, model, criterion, optimizer)
