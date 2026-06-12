FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/align

COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip install .

ENTRYPOINT ["align-pipeline"]
CMD ["--help"]
