FROM tensorflow/tensorflow:2.3.0rc1-gpu-jupyter

RUN apt-get update && apt-get install -y git
RUN mkdir /init
COPY ./requirements.txt /init/requirements.txt
RUN pip3 -q install pip --upgrade
RUN pip install -r /init/requirements.txt