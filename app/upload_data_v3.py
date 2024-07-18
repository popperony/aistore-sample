import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from aistore.pytorch import AISFileLister, AISFileLoader
from torchdata.datapipes.iter import FileLister, FileLoader, TarArchiveReader

number_of_classes = 10

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print('Загрузка данных из бакета')

bucket_name = 'imagenet-data'
lister = AISFileLister(bucket_name=bucket_name, extension_filter='.tar.gz')
loader = AISFileLoader(bucket_name=bucket_name)

files = lister.filelist()
dataset = loader.load_files(files).map(TarArchiveReader).map(lambda x: (transform(x[0]), x[1])).shuffle().batch(32)

print('Инициализация модели')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, number_of_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train_model(dataloader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss {running_loss/len(dataloader)}')

print('Подготовка данных')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print('Тренировка модели')
train_model(train_loader, model, criterion, optimizer)
