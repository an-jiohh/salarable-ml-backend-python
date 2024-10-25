FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y gcc build-essential

COPY poetry.lock pyproject.toml ./
RUN pip install poetry

# Poetry 가상 환경 비활성화 설정
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

COPY ./app ./app

RUN mkdir temp temp_image

ENTRYPOINT ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000"]