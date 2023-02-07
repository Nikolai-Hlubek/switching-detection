FROM tensorflow/tensorflow:latest-gpu-jupyter

ADD requirements.txt .

RUN pip install -r requirements.txt