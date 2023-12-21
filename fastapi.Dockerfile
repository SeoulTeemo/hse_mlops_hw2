FROM python:3.9-slim

WORKDIR /code

COPY ./pyproject.toml ./poetry.lock ./src/main.py /code
COPY ./src/app /code/app

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install

CMD poetry run python main.py

EXPOSE 8000
