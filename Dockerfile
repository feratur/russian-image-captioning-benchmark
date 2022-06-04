FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install --yes build-essential wget curl git unzip libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY ./requirements.txt ./requirements.txt
# Ugly way to install requirements because some of them are conflicting, also latest version of fairseq is not present on PyPI, so building from source
RUN git clone https://github.com/pytorch/fairseq.git && pip install ./fairseq && pip install -r requirements.txt && pip install ruclip==0.0.1 && pip install timm==0.4.12

COPY . .
RUN chmod +x evaluate.sh
CMD ["./evaluate.sh"]
