from aistore.pytorch.dataset import AISDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def get_data_loader(client_url: str, bucket_name: str, prefix: str, batch_size: int) -> DataLoader:
    """
    Функция для создания DataLoader для датасета ImageNet на базе AISTore.

    :param client_url: URL для подключения к AISTore.
    :param bucket_name: Название S3 бакета.
    :param prefix: Префикс для ключей объектов в бакете.
    :param batch_size: Размер батча.
    :return: DataLoader для обучения модели.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = AISDataset(
        client_url=client_url,
        urls_list=[f"s3://{bucket_name}/{prefix}"],
        ais_source_list=[],
        etl_name=None
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: (transform(x[0]), x[1]))
