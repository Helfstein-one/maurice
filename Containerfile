# ==========================================================
# Stage 1: Build Nativo do llama.cpp com glibc (Debian)
# ==========================================================
FROM docker.io/library/debian:bookworm-slim AS builder-native

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git

WORKDIR /src/llama.cpp
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_AVX=ON \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON
RUN cmake --build build --config Release -j$(nproc) --target llama-cli llama-quantize llama-imatrix

# ==========================================================
# Stage 2: Runtime Environment (100% glibc compatível)
# ==========================================================
FROM docker.io/library/python:3.11-slim-bookworm

LABEL maintainer="Maurício Helfstein Gonçalves"
LABEL project="maurice"

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/home/maurice/bin:/home/maurice/.local/bin:${PATH}" \
    LD_LIBRARY_PATH="/home/maurice/bin:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root para execução segura no Podman
RUN useradd -m -u 1001 -s /bin/bash maurice
USER maurice
WORKDIR /home/maurice/app

# Copia binários C++ e bibliotecas compartilhadas (.so) compilados com glibc
COPY --from=builder-native --chown=maurice:maurice /src/llama.cpp/build/bin/ /home/maurice/bin/

RUN chmod +x /home/maurice/bin/*

# Configuração de dependências Python
COPY --chown=maurice:maurice pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir --user -r requirements.txt 2>/dev/null || true

# Copia o código-fonte
COPY --chown=maurice:maurice . .

CMD ["/bin/bash"]
