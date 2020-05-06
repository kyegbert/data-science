FROM python:3.8.2-slim-buster

WORKDIR /data_science
COPY ./bin/start_jupyter.sh ./bin/start_jupyter.sh
RUN chmod +x /data_science/bin/start_jupyter.sh
COPY ./requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    graphviz libgraphviz-dev python3-pip && \
    pip install virtualenv && \
    virtualenv .venv && \
    cd .venv && \
    pip install -r /data_science/requirements.txt

EXPOSE 8888

CMD ["/data_science/bin/start_jupyter.sh"]