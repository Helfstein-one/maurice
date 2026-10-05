# 🧠 maurice
**Minimal Adaptation for Ultra-fast Reasoning and Inference in Code Engines**

[![CI Pipeline](https://github.com/Helfstein-one/maurice/actions/workflows/ci.yml/badge.svg)](https://github.com/Helfstein-one/maurice/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

maurice is an end-to-end local LLM engineering framework designed to distill, fine-tune, and align small, highly-capable SLMs (Small Language Models). By leveraging QLoRA, RLAIF (LLM-as-a-judge preference synthesis), ORPO/DPO alignment, and GGUF quantization, maurice allows anyone to build specialized AI coding engines that run locally with minimal hardware footprint.

---

## 🎯 Por Que Criar o maurice? (Vantagens & Trade-offs)

Na era de modelos monolíticos gigantes (70B+ parâmetros) hospedados em nuvem, o maurice adota a filosofia do **"Small, Specialized, and Local"**. 

### Vantagens (Por que usar?)
1. **Inferência Ultra-Rápida:** Modelos de 1.5B parâmetros quantizados em Q4_K_M entregam mais de 120+ tokens/segundo em Apple Silicon e CPUs modernas, e 200+ t/s em GPUs dedicadas.
2. **Privacidade Absoluta:** O código proprietário da sua empresa nunca sai da sua máquina. Toda a inferência (e até o treinamento) ocorre *on-premise* ou *localhost*.
3. **Especialização via RLAIF:** Em vez de tentar saber tudo, os modelos são especialistas. Se você treina a variante de código (`mau-llm-1.0-c`), a pipeline foca em alinhamento ORPO com dados curados de código, resultando em precisão cirúrgica no domínio.
4. **Baixo Custo Computacional:** A etapa de QLoRA + Flash Attention 2 exige pouquíssima VRAM (uma GPU RTX 3060 ou Mac M1 de 8GB é suficiente para fine-tuning).

### Trade-offs (O que você sacrifica?)
- **Generalização Ampla:** Sendo um modelo pequeno (SLM), ele tem menos "conhecimento de mundo" enciclopédico. É um motor de raciocínio, não um motor de busca.
- **Context Length Limits:** Embora treinado com até 4k-8k tokens de contexto, tarefas que exigem a ingestão de um repositório inteiro de 100 mil linhas sofrerão degradação (recomendamos o uso em arquiteturas RAG).
- **Risco de Alucinação em Nichos:** Sem Retrieval, o modelo tentará adivinhar bibliotecas muito obscuras. O alinhamento ORPO ajuda a mitigar isso, mas SLMs sempre performam melhor com contexto injetado (via system prompts ou embeddings).

---

## 🏗️ Arquitetura do Pipeline

A nossa pipeline é segmentada em 6 estágios modulares. Da extração do dado bruto até o binário compilado.

![Pipeline de Transformação](assets/pipeline.svg)



---

## ⚡ Performance e Benchmarks

maurice é construído para velocidade. Aqui está o perfil de inferência esperado para a família `mau-llm-1.0` (1.5B parâmetros, Q4_K_M):

![Dashboard de Benchmarks no Streamlit](images/streamlit_benchmark.jpg)

![Benchmarks de Velocidade](assets/benchmarks.svg)

| Hardware | Backend | Velocidade Esperada | VRAM Consumida |
| :--- | :--- | :--- | :--- |
| **MacBook Air M1/M2 (8GB)** | Metal (MPS) via Ollama | ~80 - 120 t/s | ~1.2 GB |
| **NVIDIA RTX 4090** | CUDA via vLLM | ~250+ t/s | ~1.2 GB |
| **x86_64 CPU (AVX2)** | llama.cpp puro | ~40 - 60 t/s | ~1.5 GB RAM |

> *A etapa de Quantização (abaixo) é o grande segredo para extrair este nível de performance na Edge.*

![Quantização Edge](assets/quantization.svg)

---

## 🛠️ Walkthrough Prático (Como usar o CLI)

A CLI do maurice (`maurice.cli`) simplifica o orquestramento. Abaixo está o fluxo completo para gerar a variante de código (`c`).

**1. Preparar o dataset (SFT)**
```bash
maurice prepare --variant c
```

**2. Treinar o Adapter QLoRA**
```bash
maurice train --variant c --batch-size 4
```

**3. Síntese RLAIF (LLM-as-a-judge)**
Gera pares de preferências (Chosen/Rejected) a partir das saídas do modelo treinado em SFT.
```bash
maurice synth-prefs --variant c
```

**4. Alinhamento Post-SFT (ORPO/DPO)**
Refina os pesos usando o dataset de preferências para desencorajar código ruim.
```bash
maurice align --variant c --method orpo
```

**5. Fazer o Merge (Consolidar pesos)**
```bash
maurice merge --variant c
```

**6. Quantizar para Edge (GGUF + imatrix)**
```bash
maurice quantize --variant c
```

---

## 🚀 Inferência Local (Ollama)

Após compilar o modelo com a pipeline acima, você pode interagir com ele nativamente usando o **Ollama**.

**1. Crie o modelo local no Ollama:**
Use o `Modelfile` preparado pela arquitetura e aponte para o arquivo GGUF gerado no diretório `build/`.
```bash
ollama create mau-llm-1.0-c -f ollama/Modelfile.c
```
*(Repita para as variantes de Raciocínio usando `ollama/Modelfile.r` e Geral com `ollama/Modelfile.g`)*

**2. Rode o modelo no seu terminal:**
O modelo agora está integrado e persistido localmente. Converse com ele diretamente via shell:
```bash
ollama run mau-llm-1.0-c "escreva um hello world em python"
```
A arquitetura do `Modelfile` já embute o *System Prompt* rigoroso, os parâmetros de *temperature* ideais de acordo com a variante e suporta conversas contínuas mantendo o contexto via formato `ChatML`.

![Model Serving API](assets/serving.svg)

---


## 🖥️ Streamlit Interactive UI

O maurice inclui uma interface gráfica (Web UI) construída em **Streamlit** para visualização e interação direta com os modelos.

![Streamlit Chat Interface](images/streamlit_chat.jpg)

```bash
# Iniciar a interface
streamlit run ui/app.py
```
Essa interface é fundamental para:
1. **Visualizar o Raciocínio (Reasoning):** Ao usar a variante `r`, a UI formata automaticamente e separa o bloco `<think>...</think>` do output final, permitindo que você entenda o processo lógico do modelo de forma didática.
2. **Avaliação Humana Rápida:** Funciona como um playground de chat interativo local.

---

## 🔄 Quickstart & Makefile Orchestration

Você também pode orquestrar todas as etapas usando o `Makefile` root:

```bash
# Executar a pipeline inteira para todas as variantes
make all

# Executar apenas para a Variante de Código 'c'
make prepare VARIANT=c
make train VARIANT=c
make synth-prefs VARIANT=c
make align VARIANT=c
make merge VARIANT=c
make quantize VARIANT=c

# Validação do código (Lint e Testes)
make lint
make test
```

---

## 🐳 Container Deployment (Podman / Docker)

```bash
# Build container image (Debian base + Glibc)
podman build -t maurice:latest -f Containerfile .

# Start FastAPI serving server inside container (Compatível com OpenAI)
podman run --rm -p 8000:8000 maurice:latest python3 scripts/06_serve_model.py
```

---

## 🛡️ CI/CD DAG Quality Gates

O repositório garante zero falhas arquiteturais usando um pipeline DAG de 6 camadas no GitHub Actions (`.github/workflows/ci.yml`), validando linting (Ruff), tipagem (Mypy), testes (Pytest) e integridade de build do container.

---

## 📚 Estrutura do Diretório

```text
maurice/
├── assets/                    # Diagramas profissionais SVG (Arquitetura & Benchmarks)
├── checkpoints/               # Artefatos intermediários (LoRA e Modelos mesclados FP16)
├── data/                      # Datasets Brutos (raw) e Processados (SFT / RLAIF)
├── maurice/                   # Core Python Package (CLI, Train, Synth, Align)
├── ollama/                    # Manifestos (Modelfiles) do Ollama para as variantes c, r, g
├── scripts/                   # Scripts standalone e servidor FastAPI vLLM
├── tests/                     # Suite de Pytest (Mocks de HuggingFace, FastAPI, etc)
├── ui/
│   └── app.py                 # Streamlit interactive visualizer & reasoning UI├── pyproject.toml             # Configuração do ecossistema via Poetry
├── Containerfile              # Runtime Glibc padronizado Docker/Podman
└── README.md                  # Esta documentação rica
```

---

## 📜 Licença e Citação

Este projeto é open-source sob a [MIT License](LICENSE).

Se o maurice foi útil nas suas pesquisas ou engenharia de IA local, considere citar:

```bibtex
@software{goncalves2026maurice,
  author = {Gon{\c{c}}alves, Maur{'i}cio Helfstein},
  title = {maurice: Minimal Adaptation for Ultra-fast Reasoning and Inference in Code Engines},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Helfstein-one/maurice}
}
```
