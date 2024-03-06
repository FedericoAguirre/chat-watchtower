FROM python:3.11.1

ENV PYTHONUNBUFFERED 1

EXPOSE 8080
WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/.
WORKDIR /app
ENV PYTHONPATH=/app

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev 
    

