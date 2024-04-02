FROM rockylinux:8.9.20231119-minimal
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN microdnf install yum
RUN yum -y install which
RUN yum -y install vi
RUN yum -y install wget
RUN yum -y update

# RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
WORKDIR /faoc
COPY . .

# ENV PROJETO="cmems"

# COPY $PROJETO /faoc/$PROJETO
# COPY environment.yml /faoc
# COPY docs /app/docs


RUN conda env create --force -f environment.yml
RUN echo "source activate cmems" > ~/.bashrc
ENV PATH /opt/conda/envs/cmems/bin:$PATH