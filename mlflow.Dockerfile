FROM python:3.9-slim

RUN pip install mlflow boto3 psycopg2-binary
