# CPU image: the release is verified on CPU, and the simulator is the only heavy component.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/respire/src

WORKDIR /opt/respire

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE NOTICE dataset_urls.txt ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python3", "-m", "respire.cli.train"]
CMD ["--config", "configs/round/_smoke.yaml", "--output", "/tmp/respire"]
