import io
import os
import pickle

import pandas as pd
from fastapi import HTTPException
from minio import Minio


# Get minio env variables
minio_host = os.getenv('MINIO_HOST')
minio_port = os.getenv('MINIO_PORT')

if minio_host is None:
    minio_host = 'minio'
if minio_port is None:
    minio_port = 9000

minio_url = f'{minio_host}:{minio_port}'

# Get env minio user and password
minio_user = os.getenv('MINIO_USER')
minio_password = os.getenv('MINIO_PASSWORD')

# Set up the client
MinioClient = Minio(
    minio_url,
    access_key=minio_user,
    secret_key=minio_password,
    secure=False
)

# MinioClient = Minio(
#     'localhost:9000',
#     access_key='lolkekcheburek',
#     secret_key='lolkekcheburek',
#     secure=False
# )


def create_bucket_if_not_exists(bucket_name: str):
    """
    Create a bucket in Minio if it does not exist
    :param bucket_name: name of the bucket
    :return: None
    """
    if not MinioClient.bucket_exists(bucket_name):
        MinioClient.make_bucket(bucket_name)
    return


def get_objects_in_bucket(bucket_name: str):
    """
    Returns list of object names in the bucket
    :param bucket_name: name of the bucket
    :return: list of objects names in the bucket
    """
    if MinioClient.bucket_exists(bucket_name):
        bucket_objects = MinioClient.list_objects(bucket_name)
        available_objects = [obj.object_name for obj in bucket_objects]
    else:
        MinioClient.make_bucket(bucket_name)
        available_objects = []
    return available_objects


def load_dataset_from_minio(
        dataset_name: str,
        target_col: str,
        bucket_name: str = 'datasets'
):
    """
    Loads dataset from minio as pandas dataframe
    :param bucket_name: name of the bucket
    :param dataset_name: name of the dataset to load
    :param target_col: columns from the dataset to use as a target
    :return: X, y
    """
    # Get dataset
    obj = MinioClient.get_object(
        bucket_name,
        dataset_name
    )
    # Read data and decode
    data = obj.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(data), sep=',')

    # Check the target column
    if target_col not in df.columns.tolist():
        raise HTTPException(status_code=404, detail=f'There is no such column {target_col} in the dataset')

    # Get target column
    y = df[target_col]
    # Get only numeric columns (no feature preprocessing for categorical columns yet)
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=['int', 'float'])

    return X, y


def save_model_to_minio(
        model,
        model_name_to_save: str,
        bucket_name: str = 'models'
):
    """
    Save the model to Minio
    :param model: model to save
    :param model_name_to_save: name of the model to be saved
    :param bucket_name: name of the bucket
    :return: None
    """
    model_obj = pickle.dumps(model)
    MinioClient.put_object(
        bucket_name=bucket_name,
        object_name=model_name_to_save,
        data=io.BytesIO(model_obj),
        length=len(model_obj)
    )
    return


def load_model_from_minio(
        model_name: str,
        bucket_name: str = 'models'
):
    """
    Loads the model from minio
    :param model_name: name of the model to load
    :param bucket_name: name of the bucket
    :return: loaded model
    """
    clf = pickle.loads(
        MinioClient.get_object(
            bucket_name=bucket_name,
            object_name=model_name
        ).read()
    )
    return clf
