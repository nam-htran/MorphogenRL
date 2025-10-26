FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    swig \
    cmake \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /kaggle/working

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
