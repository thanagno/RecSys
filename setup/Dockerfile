# Use an official Python runtime as the base image
# --- pull official base image --- #
ARG BASEIMAGE=python:3.11-slim

FROM $BASEIMAGE as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV JUPYTER_PORT 8989

# ------ Set up workspace -------- #
RUN true && apt-get update
RUN apt update
RUN apt-get install -y curl
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y gcc

## --- upgrade pip --- #
RUN true \
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade pip setuptools

# Set the working directory in the container
RUN mkdir /workspace

# --- Set Workspace --- #
WORKDIR "/workspace"

# Copy the requirements.txt file into the container and Install the required Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

 # It is better to do do it using a setup.py file
 COPY setup.py .
# RUN pip install --no-cache-dir .

# Copy the 'source' and 'data' directories into the container
COPY ./src ./src
COPY ./data ./data
COPY ./notebooks ./notebooks
COPY ./tests ./tests
