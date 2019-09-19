FROM ubuntu:18.04

# install basics
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing \
    && apt-get install -y \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        fonts-noto-cjk \
        git \
        htop \
        locales \
        sudo \
        tmux \
        tree \
        tzdata \
        unzip \
        vim \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# set locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# install conda
ENV PATH /opt/conda/bin:$PATH
ARG miniconda_dl_path="https://repo.continuum.io/miniconda"
ARG miniconda="Miniconda3-4.7.10-Linux-x86_64.sh"
RUN wget --quiet $miniconda_dl_path/$miniconda -O /root/miniconda.sh \
    && mkdir /root/.conda/ \
    && /bin/bash /root/miniconda.sh -b -p /opt/conda \
    && rm /root/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
    && echo "conda activate base" >> /root/.bashrc \
    && find /opt/conda/ -follow -type f -name "*.a" -delete \
    && find /opt/conda/ -follow -type f -name "*.js.map" -delete \
    && /opt/conda/bin/conda clean -afy \
    && conda update conda \
    && conda init bash

# create conda environment
COPY ./environment.yml /environment.yml
RUN conda env create -f /environment.yml -n env \
    && echo "conda activate env" >> /root/.bashrc

# set config
ENV TZ Asia/Tokyo
WORKDIR /work
