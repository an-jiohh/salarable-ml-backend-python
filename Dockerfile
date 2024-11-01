FROM python:3.11.9-slim

ENV TZ=Asia/Seoul

COPY requirements.txt  ./

RUN pip install -r requirements.txt

COPY ./app ./app

RUN mkdir temp temp_image

ENTRYPOINT ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000"]