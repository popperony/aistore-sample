import torch.nn as nn
import torchvision.models as models


def get_resnet18_model(pretrained: bool = False) -> nn.Module:
    """
    Функция для получения модели ResNet-18.

    :param pretrained: Использовать предобученную модель или нет.
    :return: Модель ResNet-18.
    """
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1000)
    return model
