import os
from datasets import load_dataset
from aistore import Client


def download_and_upload_data(bucket_name: str, prefix: str) -> None:
    """
    Функция для загрузки датасета ImageNet с Hugging Face и его загрузки в S3 через aistore.

    :param bucket_name: Название S3 бакета.
    :param prefix: Префикс для ключей объектов в бакете.
    """
    dataset = load_dataset("ILSVRC/imagenet-1k", split='train')

    client = Client("http://localhost:51080/")

    temp_dir = "data/ilsvrc_imagenet1k/"
    os.makedirs(temp_dir, exist_ok=True)

    for idx, sample in enumerate(dataset):
        image = sample['image']

        image_path = os.path.join(temp_dir, f"train_{idx}.jpg")
        image.save(image_path)

        object_name = f"{prefix}/train/{idx}.jpg"
        with open(image_path, 'rb') as f:
            client.put_object(bucket_name, object_name, f)

        os.remove(image_path)

        print(f"Uploaded {object_name}")


if __name__ == "__main__":
    bucket_name = "imagenet-data"
    prefix = "images"
    download_and_upload_data(bucket_name, prefix)
