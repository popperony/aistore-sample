import aistore.sdk as ais
import time


def load_dataset_from_url(aistore_url: str, dataset_url: str, aistore_bucket_name: str, object_name: str) -> None:
    client = ais.Client(aistore_url)

    buckets = client.cluster().list_buckets()
    if aistore_bucket_name not in [b.name for b in buckets]:
        client.bucket(aistore_bucket_name).create()
        print(f"Bucket {aistore_bucket_name} created in AIStore.")
    else:
        print(f"Bucket {aistore_bucket_name} already exists in AIStore.")

    start_time = time.time()

    try:
        response = client.bucket(aistore_bucket_name).put_object(
            obj_name=object_name,
            data=dataset_url,
            params={"source": "web"}
        )

        if response.status_code == 200:
            print(f"File {object_name} downloaded to AIStore bucket {aistore_bucket_name} successfully.")
        else:
            raise Exception(f"Failed to download file to AIStore. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken to load dataset: {total_time:.2f} seconds")


if __name__ == "__main__":
    AISTORE_URL = "http://localhost:51080"
    AISTORE_BUCKET_NAME = 'imagenet-data'
    S3_BUCKET_NAME = "imagenet-data"
    DATASET_URL = 'https://huggingface.co/datasets/ILSVRC/imagenet-1k/blob/main/data/train_images_0.tar.gz'
    OBJECT_NAME = 'train_images_0.tar.gz'

    load_dataset_from_url(AISTORE_URL, DATASET_URL, AISTORE_BUCKET_NAME, OBJECT_NAME)
