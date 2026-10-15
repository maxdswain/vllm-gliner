ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.31.0
FROM ${VLLM_BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/maxdswain/vllm-gliner"

WORKDIR /opt/vllm-gliner
COPY pyproject.toml README.md LICENSE ./
COPY forge/ forge/
COPY kernels/ kernels/
COPY models/ models/
COPY plugins/ plugins/
COPY poolers/ poolers/
COPY vllm_factory/ vllm_factory/
RUN pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir 'gliner2==2.1.0' sentencepiece
