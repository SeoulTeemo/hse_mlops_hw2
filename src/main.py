import io
import os
from typing import Union

from fastapi import FastAPI, File
from fastapi import HTTPException
from minio import Minio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from typing_extensions import Annotated

from app.models_setup import ModelName, DecisionTreeParams, LogisticRegressionParams
from app.utils import create_bucket_if_not_exists, get_objects_in_bucket, load_dataset_from_minio, load_model_from_minio, \
    save_model_to_minio

app = FastAPI()

MODELS_BUCKET_NAME = 'models'
DATASETS_BUCKET_NAME = 'datasets'

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


@app.get("/")
async def root():
    # Some 'Hello world!' like message
    return {"message": "Hello, my Lord"}


@app.post("/available_model_classes")
async def get_model_classes_available():
    """
    Shows available model classes to train.
    :return: list of available models' classes
    """
    # Get all models classes as a dictionary
    models_classes = {f'Model_{idx}': model.value for (idx, model) in enumerate(ModelName)}
    return models_classes


@app.post("/available_datasets")
async def get_available_datasets_list():
    """
    Shows uploaded datasets for training.
    :return: list of available models' classes
    """

    # Get all the datasets in the datasets' bucket
    available_datasets = get_objects_in_bucket(DATASETS_BUCKET_NAME)
    num_datasets = len(available_datasets)
    return {"List of available datasets": available_datasets, 'Overall number of uploaded datasets': num_datasets}


@app.post("/fitted_models")
async def get_fitted_models_list():
    """
    Returns fitted models.
    :return:
    """

    # Get all objects in the models bucket
    available_fitted_models = get_objects_in_bucket(MODELS_BUCKET_NAME)
    num_models = len(available_fitted_models)
    return {"List of available fitted models": available_fitted_models, 'Overall number of fitted models': num_models}


@app.post("/upload_dataset/")
async def upload_dataset(
        dataset_file: Annotated[bytes, File()],
        dataset_name: str
):
    """
    Uploads dataset in the format of a csv file
    :param dataset_file: dataset file.
    :param dataset_name: name of the dataset
    :return:
    """

    # Check that the bucket exists
    create_bucket_if_not_exists(DATASETS_BUCKET_NAME)

    # Sabe the dataset
    MinioClient.put_object(
        bucket_name=DATASETS_BUCKET_NAME,
        object_name=dataset_name,
        data=io.BytesIO(dataset_file),
        length=len(dataset_file),
        content_type='application/xml'
    )

    return {'message': f'The dataset {dataset_name} has been uploaded.'}


@app.post("/fit_model")
async def fit_model(
        clf_name: ModelName,
        dataset_name: str,
        target_col: str,
        model_name_to_save: str,
        params: Union[DecisionTreeParams, LogisticRegressionParams, None] = None
):
    """
    Fits a model
    :param clf_name: model class of the model
    :param dataset_name: name of the dataset
    :param target_col: column from the dataset to use as a target
    :param model_name_to_save: name of the model to be fitted
    :param params: hyperparameters of the model
    :return:
    """
    # Check there are some params given
    if params is not None:
        hyperparams = params.model_dump()
    else:
        hyperparams = {}

    # Get classifier class
    if clf_name is ModelName.decision_tree:
        model = DecisionTreeClassifier(**hyperparams)
    elif clf_name is ModelName.logistic_regression:
        model = LogisticRegression(**hyperparams)
    else:
        raise HTTPException(status_code=404, detail='Such model is not supported')

    # Check that hyperparams are valid
    if len(hyperparams) != 0:
        # Get params
        all_model_params_available = list(model.get_params().keys())
        params_given = list(params.model_fields.keys())
        for param in params_given:
            if param not in all_model_params_available:
                raise HTTPException(status_code=404, detail=f'Found unknown hyperparameter {param}')

    # Load the dataset and fit the model
    X, y = load_dataset_from_minio(dataset_name, target_col)
    X = X.fillna(0)
    clf = model.fit(X, y)

    # Save the model
    create_bucket_if_not_exists(MODELS_BUCKET_NAME)
    save_model_to_minio(clf, model_name_to_save, bucket_name=MODELS_BUCKET_NAME)

    return {"message": f"The model {model_name_to_save} has been trained and saved successfully"}


@app.post("/refit_model")
async def refit_model(
        model_name: str,
        dataset_name: str,
        target_col: str
):
    """
    Refits the existing model with the given dataset
    :param model_name: name of the model to refit
    :param dataset_name: name of the dataset to fit the model on
    :param target_col: columns from the dataset to use as a target
    :return: Refitted model
    """

    # Check if the bucket exists
    if not MinioClient.bucket_exists(MODELS_BUCKET_NAME):
        raise HTTPException(status_code=404, detail=f'There is no such a bucket {MODELS_BUCKET_NAME}')

    # Check if the model is fitted
    available_fitted_models = get_objects_in_bucket(MODELS_BUCKET_NAME)
    if model_name not in available_fitted_models:
        raise HTTPException(status_code=404, detail=f'There is no such a fitted model {model_name}')

    # Load the model
    clf = load_model_from_minio(model_name, bucket_name=MODELS_BUCKET_NAME)

    # Get old model params
    new_model_params = dict(clf.get_params())
    if type(clf).__name__ == 'DecisionTreeClassifier':
        new_model = DecisionTreeClassifier(**new_model_params)
    else:
        new_model = LogisticRegression(**new_model_params)

    # Get dataset and fit the model
    X, y = load_dataset_from_minio(dataset_name, target_col)
    X = X.fillna(0)
    clf = new_model.fit(X, y)

    # Remove the old model
    MinioClient.remove_object(MODELS_BUCKET_NAME, model_name)
    # Save the new model
    save_model_to_minio(clf, model_name, bucket_name=MODELS_BUCKET_NAME)

    return {"message": "The model has been trained and saved successfully"}


@app.post("/predict/{model_name}")
async def model_predict(
        model_name: str,
        dataset_name: str,
        target_col: str,
):
    """
    Predict proba with the given model
    :param model_name:
    :param dataset_name:
    :param target_col:
    :return:
    """

    # Check if the bucket exists
    if not MinioClient.bucket_exists(MODELS_BUCKET_NAME):
        raise HTTPException(status_code=404, detail=f'There is no such a bucket {MODELS_BUCKET_NAME}')

    # Check if the model is fitted
    available_fitted_models = get_objects_in_bucket(MODELS_BUCKET_NAME)
    if model_name not in available_fitted_models:
        raise HTTPException(status_code=404, detail=f'There is no such a fitted model {model_name}')

    # Get dataset
    X, y_true = load_dataset_from_minio(dataset_name, target_col)
    X = X.fillna(0)
    # Load the model
    clf = load_model_from_minio(model_name, bucket_name=MODELS_BUCKET_NAME)

    # Get y_proba
    y_pred_proba = clf.predict_proba(X)[:, 1]
    # Calculate some metric
    auc = roc_auc_score(y_true, y_pred_proba)

    return {"roc_auc_score": auc, "prediction_proba": y_pred_proba.tolist()}


@app.post("/remove/{model_name}")
async def delete_model(model_name: str):
    """
    Deletees model from minio
    :param model_name: name of the model to delete
    :return:
    """
    # Check if the bucket exists
    if not MinioClient.bucket_exists(MODELS_BUCKET_NAME):
        raise HTTPException(status_code=404, detail=f'There is no such a bucket {MODELS_BUCKET_NAME}')

    # Check if the model is fitted
    available_fitted_models = get_objects_in_bucket(MODELS_BUCKET_NAME)
    if model_name not in available_fitted_models:
        raise HTTPException(status_code=404, detail=f'There is no such a fitted model {model_name}')

    MinioClient.remove_object(MODELS_BUCKET_NAME, model_name)
    return {'Result': f'The model {model_name} was removed successfully'}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    print('The main.py has been run.')
