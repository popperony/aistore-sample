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

    temp_dir = "data/ilsvrc_imagenet1k/"
    os.makedirs(temp_dir, exist_ok=True)
    print('Start upload dataset...')

    for idx, sample in enumerate(dataset):
        image = sample['image']

        image_path = os.path.join(temp_dir, f"train_{idx}.jpg")
        image.save(image_path)

        try:
            object_name = f"{prefix}/train/{idx}.jpg"
            with open(image_path, 'rb') as f:
                bucket.object(object_name).put(f.read())

            print(f"Uploaded {object_name}")
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted temporary file {image_path}")
            else:
                print(f"File {image_path} does not exist")


if __name__ == "__main__":
    client_url = "http://localhost:51080"
    bucket_name = "imagenet-data"
    prefix = "images"
    hf_token = "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"
    download_and_upload_data(client_url, bucket_name, prefix, hf_token)
