.PHONY: all prepare train merge quantize eval serve code-review ui clean help test dry-run lint quality-gates mcp-server validate-reasoning

maurice ?= maurice
PYTHON ?= python3
PORT ?= 8000
VARIANT ?= c

all: prepare train merge quantize eval

prepare:
	$(maurice) prepare --variant $(VARIANT)

train:
	@if [ "$(VARIANT)" = "all" ]; then \
		$(maurice) train --variant c; \
		$(maurice) train --variant r; \
		$(maurice) train --variant g; \
	else \
		$(maurice) train --variant $(VARIANT); \
	fi

merge:
	@if [ "$(VARIANT)" = "all" ]; then \
		$(maurice) merge --variant c; \
		$(maurice) merge --variant r; \
		$(maurice) merge --variant g; \
	else \
		$(maurice) merge --variant $(VARIANT); \
	fi

quantize:
	@if [ "$(VARIANT)" = "all" ]; then \
		$(maurice) quantize --variant c; \
		$(maurice) quantize --variant r; \
		$(maurice) quantize --variant g; \
	else \
		$(maurice) quantize --variant $(VARIANT); \
	fi

eval:
	$(maurice) eval --variant $(VARIANT)

validate-reasoning:
	$(PYTHON) scripts/validate_reasoning.py --dry-run

serve:
	$(maurice) serve --variant $(VARIANT) --port $(PORT)

code-review:
	$(PYTHON) scripts/08_code_review.py

dry-run:
	$(maurice) prepare --variant all --dry-run
	$(maurice) train --variant c --dry-run
	$(maurice) merge --variant c --dry-run
	$(maurice) eval --variant all --dry-run
	$(PYTHON) scripts/validate_reasoning.py --dry-run
	$(PYTHON) -m py_compile scripts/06_serve_model.py
	$(PYTHON) -m py_compile scripts/07_publish_hub.py
	$(PYTHON) -m py_compile scripts/08_code_review.py
	$(PYTHON) -m py_compile scripts/08_mcp_quality_gates.py

quality-gates:
	$(PYTHON) scripts/08_mcp_quality_gates.py --run-gates

mcp-server:
	$(PYTHON) scripts/08_mcp_quality_gates.py --port $(or $(PORT),8080)

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .
	ruff format --check .

ui:
	$(maurice) ui

clean:
	rm -rf checkpoints/ build/ results/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name '*.pyc' -delete

help:
	@echo 'maurice Pipeline Makefile'
	@echo ''
	@echo 'Targets:'
	@echo '  all        Run full pipeline (prepare → train → merge → quantize → eval)'
	@echo '  prepare    Prepare datasets for all variants'
	@echo '  train      Train QLoRA adapters (c, r, g)'
	@echo '  merge      Merge adapter weights into base model'
	@echo '  quantize   Convert to GGUF and quantize with imatrix'
	@echo '  eval       Run benchmark evaluation'
	@echo '  validate-reasoning Validate mau-llm-1.0-r mathematical & CoT reasoning'
	@echo '  serve      Launch FastAPI inference server'
	@echo '  code-review Run proactive code review on a PR'
	@echo '  dry-run    Smoke test entire pipeline without GPU'
	@echo '  test          Run pytest suite'
	@echo '  lint          Run ruff linter'
	@echo '  quality-gates Run quality & validation gates locally'
	@echo '  mcp-server    Start local MCP Quality Gates server'
	@echo '  ui         Run Streamlit UI'
	@echo '  clean      Remove build artifacts'
