import io
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from aistore import Client
from io import BytesIO
import tarfile
from PIL import Image


client = Client("http://localhost:51080")


bucket_name = "imagenet-data"

print('Загрузка или создание бакета')
bucket = client.bucket(bucket_name)
bucket.create(exist_ok=True)

print('Загрузка файла в бакет')
local_file_path = "/home/test/train_images_0.tar.gz"
object_name = "train_images_0.tar.gz"

with open(local_file_path, "rb") as file:
    bucket.object(object_name).put_file(local_file_path)


def load_data_from_tar(tar_content):
    tar = tarfile.open(fileobj=tar_content)
    images = []
    labels = []
    for member in tar.getmembers():
        if member.name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
            class_id = member.name.split('/')[0]

            f = tar.extractfile(member)
            if f:
                img = Image.open(io.BytesIO(f.read()))
                img = img.convert('RGB')
                images.append(transforms.ToTensor()(img))
                labels.append(class_id)

    if not images:
        raise ValueError("Не найдено подходящих изображений в архиве")

    return torch.stack(images), labels


try:
    object_reader = bucket.object(object_name).get()
    object_content = io.BytesIO()
    for chunk in object_reader:
        object_content.write(chunk)
    object_content.seek(0)
except Exception as e:
    print(f"Ошибка при получении объекта {object_name}: {str(e)}")
    raise

print('Обработка данных')
train_images, train_labels = load_data_from_tar(object_content)

print('Подготовка датасета')
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)

print('Подготовка данных')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


print('Инициализация resnet')
model = models.resnet18(pretrained=False)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start_time = time.time()
print('Запуск обучения')
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

end_time = time.time()
total_time = end_time - start_time

print(f"Общее время обучения: {total_time:.2f} секунд")
