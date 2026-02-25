FROM python:3.12-slim

ARG MLFLOW_ARTFACT_STORE


RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip install --no-cache-dir mlflow psycopg2-binary

COPY ./docker/run-server.sh /run-server.sh

RUN chmod +x /run-server.sh

CMD ["/run-server.sh"]