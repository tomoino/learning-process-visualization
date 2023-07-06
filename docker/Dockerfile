FROM python:3.10-slim

ENV WORK_PATH /workspace

RUN apt-get update && \
    apt-get install -y \
			sudo curl git build-essential tmux python3-pip && \
    apt-get clean && \
		rm -rf /var/lib/apt/lists/* && \
		rm -rf /var/cache/apk/

COPY requirements.txt .
ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"
RUN pip3 install -U pip && \
    pip3 install -r requirements.txt

WORKDIR /workspace