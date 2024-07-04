import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import get_resnet18_model
from data_loader import get_data_loader


def train_model(data_loader: DataLoader, model: nn.Module, num_epochs: int = 10) -> None:
    """
    Функция для обучения модели.

    :param data_loader: DataLoader с обучающими данными.
    :param model: Модель для обучения.
    :param num_epochs: Количество эпох обучения.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(data_loader)}")


if __name__ == "__main__":
    client_url = "http://localhost:51080/"
    bucket_name = "imagenet-data"
    prefix = "images/train"
    data_loader = get_data_loader(client_url=client_url, bucket_name=bucket_name, prefix=prefix, batch_size=32)
    model = get_resnet18_model(pretrained=True)
    train_model(data_loader, model, num_epochs=10)
