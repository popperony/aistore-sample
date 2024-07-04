import boto3
import time

def upload_to_s3(bucket_name, file_path, object_name=None):
    s3_client = boto3.client('s3')
    if object_name is None:
        object_name = file_path
    start_time = time.time()
    s3_client.upload_file(file_path, bucket_name, object_name)
    end_time = time.time()
    return end_time - start_time

# Пример использования
bucket_name_ceph = 'my-ceph-bucket'
bucket_name_ais = 'my-ais-bucket'
file_path = 'path/to/imagenet.zip'

# Загрузка в Ceph
time_ceph = upload_to_s3(bucket_name_ceph, file_path)
print(f"Upload to Ceph: {time_ceph} seconds")

# Загрузка в AIStore
time_ais = upload_to_s3(bucket_name_ais, file_path)
print(f"Upload to AIStore: {time_ais} seconds")
