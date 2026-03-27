FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update &&\
    apt-get install -y sudo wget unzip vim nano python3 python3-pip python3-venv tzdata libgmp3-dev git
RUN wget -O - https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install_prereqs.sh | bash &&\
    wget -O - https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | bash

COPY ./requirements.txt ./lbp-neural-cbf/requirements.txt

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install -r lbp-neural-cbf/requirements.txt

COPY . ./lbp-neural-cbf
WORKDIR /lbp-neural-cbf

RUN pip install -e .
RUN pip install -e ./fossil

ENTRYPOINT [ "bash" ]