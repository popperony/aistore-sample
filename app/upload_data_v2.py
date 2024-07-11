import os
import tarfile
import time

import requests
import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from aistore.pytorch import AISMapDataset
from aistore.sdk import Client
from torch.utils.data import DataLoader

dataset_url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/train_images_0.tar.gz"
dataset_path = "train_images_0.tar.gz"
extracted_path = "train_images"

start_time_download = time.time()
if not os.path.exists(dataset_path):
    response = requests.get(dataset_url)
    with open(dataset_path, 'wb') as f:
        f.write(response.content)
end_time_download = time.time()

start_time_extraction = time.time()
if not os.path.exists(extracted_path):
    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall(extracted_path)
end_time_extraction = time.time()

start_time_upload = time.time()
ais_url = os.getenv("AIS_ENDPOINT", "http://localhost:51080")
client = Client(ais_url)
bucket_name = "imagenet-data1"
bucket = client.bucket(bucket_name).create(exist_ok=True)

for root, _, files in os.walk(extracted_path):
    for file in files:
        file_path = os.path.join(root, file)
        with open(file_path, 'rb') as f:
            bucket.object(file).put(f)
end_time_upload = time.time()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ais_dataset = AISMapDataset(
    ais_source_list=[bucket],
    transform=transform
)

start_time_loading = time.time()
data_loader = DataLoader(
    ais_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
end_time_loading = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 5

start_time_training = time.time()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0
end_time_training = time.time()

print("Обучение завершено")

print(f"Время загрузки датасета: {end_time_download - start_time_download:.2f} секунд")
print(f"Время распаковки датасета: {end_time_extraction - start_time_extraction:.2f} секунд")
print(f"Время загрузки данных в AIStore: {end_time_upload - start_time_upload:.2f} секунд")
print(f"Время создания DataLoader: {end_time_loading - start_time_loading:.2f} секунд")
print(f"Время обучения модели: {end_time_training - start_time_training:.2f} секунд")
