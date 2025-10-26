FROM gcr.io/kaggle-images/python:latest
COPY . .
RUN pip install "gymnasium==0.28.1" "ray[rllib]"
