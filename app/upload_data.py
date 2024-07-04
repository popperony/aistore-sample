import io

from aistore import Client
from datasets import load_dataset
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
    bucket.create(exist_ok=True)

    for id_, sample in enumerate(dataset):
        image = sample['image']
        label = sample['label']

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        object_name = f"{prefix}/train/{label}/{id_}.jpg"

        bucket.object(object_name).put(data=img_byte_arr.getvalue(), content_type='image/jpeg')

        print(f"Uploaded {object_name}")


if __name__ == "__main__":
    client_url = "http://localhost:51080"
    bucket_name = "imagenet-data"
    prefix = "images"
    hf_token = "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"
    download_and_upload_data(client_url, bucket_name, prefix, hf_token)
