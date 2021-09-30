ARG NVIDIA_CONTAINER=nvcr.io/nvidia/tensorrt:21.08-py3
FROM $NVIDIA_CONTAINER

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install python3-opencv -y