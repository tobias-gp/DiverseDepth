FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel


WORKDIR /app

RUN apt-get -y update && apt-get -y install \
        libglib2.0-0 \
        libsm6 \
        libxrender-dev \
        libxext6 \
        unzip \
        wget \
        nano 

RUN nvidia-smi

RUN conda update -n base -c defaults conda

RUN conda install python=3.6

RUN conda install -c pytorch torchvision=0.2.1 cudatoolkit=9.0

RUN conda install matplotlib dill pyyaml opencv scipy 

RUN pip install --upgrade pip

RUN wget -O model.pth https://cloudstor.aarnet.edu.au/plus/s/ixWf3nTJFZ0YE4q/download

ENV PYTHONUNBUFFERED=.

# Finally copy all files over
COPY . . 

ENTRYPOINT ["python", "tools/inference.py"]
