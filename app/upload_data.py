import os
from datasets import load_dataset
from aistore import Client


def download_and_upload_data(client_url: str, bucket_name: str, prefix: str) -> None:
    """
    Функция для загрузки датасета ImageNet с Hugging Face и его загрузки в S3 через aistore.

    :param client_url: URL для подключения к AISTore.
    :param bucket_name: Название S3 бакета.
    :param prefix: Префикс для ключей объектов в бакете.
    """
    dataset = load_dataset("ILSVRC/imagenet-1k", split='train')

    client = Client(client_url)
    bucket = client.bucket(bucket_name)

    temp_dir = "data/ilsvrc_imagenet1k/"
    os.makedirs(temp_dir, exist_ok=True)

    for idx, sample in enumerate(dataset):
        image = sample['image']

        image_path = os.path.join(temp_dir, f"train_{idx}.jpg")
        image.save(image_path)

        object_name = f"{prefix}/train/{idx}.jpg"
        bucket.object(object_name).put(image_path)

        os.remove(image_path)

        print(f"Uploaded {object_name}")


if __name__ == "__main__":
    client_url = "http://localhost:51080/"
    bucket_name = "imagenet-data"
    prefix = "images"
    download_and_upload_data(client_url, bucket_name, prefix)
