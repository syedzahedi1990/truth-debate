FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/truth-debate
COPY requirements-vast.txt requirements.txt ./
RUN python -m pip install --upgrade pip wheel setuptools \
    && python -m pip install -r requirements-vast.txt

COPY . .
RUN python -m pip install -e .

CMD ["bash", "scripts/run_end_to_end.sh", "configs/vast_0_5b.yaml"]
