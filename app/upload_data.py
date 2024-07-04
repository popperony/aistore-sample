import os
from datasets import load_dataset
from aistore import Client
from huggingface_hub import login


def download_and_upload_data(client_url: str, bucket_name: str, prefix: str, hf_token: str) -> None:
    """
    Функция для загрузки датасета ImageNet с Hugging Face и его загрузки в S3 через aistore.

    :param client_url: URL для подключения к AISTore.
    :param bucket_name: Название S3 бакета.
    :param prefix: Префикс для ключей объектов в бакете.
    :param hf_token: Токен аутентификации Hugging Face.
    """
    login(hf_token)
    print('Start downloading data...')
    dataset = load_dataset("ILSVRC/imagenet-1k", split='train', streaming=True)

    client = Client(client_url)
    bucket = client.bucket(bucket_name)

    for id_, sample in enumerate(dataset):
        image = sample['image']

        image_path = f"temp_train_{id_}.jpg"
        image.save(image_path)

        object_name = f"{prefix}/train/{id_}.jpg"
        with open(image_path, 'rb') as f:
            print(f'Uploading {object_name}')
            bucket.put_object(object_name, data=f, content_type='image/jpeg')

        os.remove(image_path)


if __name__ == "__main__":
    client_url = "http://localhost:51080"
    bucket_name = "imagenet-data"
    prefix = "images"
    hf_token = "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"
    download_and_upload_data(client_url, bucket_name, prefix, hf_token)
