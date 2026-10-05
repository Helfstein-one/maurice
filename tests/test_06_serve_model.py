"""
tests/test_06_serve_model.py
Comprehensive unit and integration test suite for scripts/06_serve_model.py FastAPI inference server.
"""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

script_path = Path(__file__).parent.parent / "scripts" / "06_serve_model.py"
spec = importlib.util.spec_from_file_location("serve_model", script_path)
serve_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_model)


@pytest.fixture(autouse=True)
def configure_mock_server():
    """Configures server_state to use mock backend and default variant 'c' for fast testing."""
    original_backend = serve_model.server_state.backend
    original_variant = serve_model.server_state.variant
    original_model = serve_model.server_state.model
    original_tokenizer = serve_model.server_state.tokenizer

    serve_model.server_state.backend = "mock"
    serve_model.server_state.variant = "c"
    serve_model.server_state.model = None
    serve_model.server_state.tokenizer = None

    yield

    serve_model.server_state.backend = original_backend
    serve_model.server_state.variant = original_variant
    serve_model.server_state.model = original_model
    serve_model.server_state.tokenizer = original_tokenizer


def test_format_chatml_prompt():
    messages = [
        serve_model.ChatMessage(role="user", content="Hello, maurice!"),
    ]
    prompt = serve_model.format_chatml_prompt(messages, "c")
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user\nHello, maurice!<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_format_chatml_prompt_custom_system():
    messages = [
        serve_model.ChatMessage(role="system", content="Custom system prompt"),
        serve_model.ChatMessage(role="user", content="Hello!"),
    ]
    prompt = serve_model.format_chatml_prompt(messages, "g")
    assert "<|im_start|>system\nCustom system prompt<|im_end|>" in prompt
    assert prompt.count("<|im_start|>system") == 1


def test_health_endpoint():
    with TestClient(serve_model.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["variant"] == "c"
        assert data["backend"] == "mock"


def test_models_endpoint():
    with TestClient(serve_model.app) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "mau-llm-1.0-c"


def test_chat_completions_non_streaming():
    with TestClient(serve_model.app) as client:
        payload = {
            "model": "mau-llm-1.0-c",
            "messages": [{"role": "user", "content": "Write python code"}],
            "temperature": 0.5,
            "max_tokens": 128,
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0


def test_chat_completions_streaming():
    with TestClient(serve_model.app) as client:
        payload = {
            "model": "mau-llm-1.0-c",
            "messages": [{"role": "user", "content": "Explain SSE streaming"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [line.strip() for line in response.text.split("\n\n") if line.strip()]
        assert len(lines) > 1

        chunks = []
        has_done = False

        for line in lines:
            if line == "data: [DONE]":
                has_done = True
                continue
            assert line.startswith("data: ")
            data = json.loads(line[6:])
            assert data["object"] == "chat.completion.chunk"
            chunks.append(data)

        assert has_done
        assert len(chunks) > 0


def test_chat_completions_validation_error_empty_messages():
    with TestClient(serve_model.app) as client:
        payload = {
            "model": "mau-llm-1.0-c",
            "messages": [],
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422


def test_chat_completions_validation_error_invalid_temperature():
    with TestClient(serve_model.app) as client:
        payload = {
            "model": "mau-llm-1.0-c",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 5.0,  # exceeds le=2.0
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422


def test_cli_parse_args():
    test_args = [
        "06_serve_model.py",
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
        "--variant",
        "r",
        "--backend",
        "mock",
    ]
    with patch("sys.argv", test_args):
        args = serve_model.parse_args()
        assert args.host == "127.0.0.1"
        assert args.port == 9000
        assert args.variant == "r"
        assert args.backend == "mock"


def test_cli_parse_args_engine_vllm():
    test_args = [
        "06_serve_model.py",
        "--variant",
        "r",
        "--engine",
        "vllm",
    ]
    with patch("sys.argv", test_args):
        args = serve_model.parse_args()
        assert args.variant == "r"
        assert args.engine == "vllm"


def test_vllm_chat_completions_mock_fallback():
    serve_model.server_state.engine = "vllm"
    serve_model.server_state.backend = "vllm"
    serve_model.server_state.vllm_engine = None

    with TestClient(serve_model.app) as client:
        payload = {
            "model": "mau-llm-1.0-r",
            "messages": [{"role": "user", "content": "Test reasoning"}],
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "vLLM" in data["choices"][0]["message"]["content"] or "mock" in data["choices"][0]["message"]["content"]


def test_vllm_engine_generation_mocked():
    serve_model.server_state.engine = "vllm"
    serve_model.server_state.backend = "vllm"

    class MockOutputText:
        def __init__(self, text: str):
            self.text = text

    class MockRequestOutput:
        def __init__(self, text: str):
            self.outputs = [MockOutputText(text)]

    class MockVLLMEngine:
        async def generate(self, prompt, sampling_params, request_id):
            yield MockRequestOutput("step 1: reasoning ")
            yield MockRequestOutput("step 1: reasoning step 2: result")

    with TestClient(serve_model.app) as client:
        serve_model.server_state.vllm_engine = MockVLLMEngine()
        payload = {
            "model": "mau-llm-1.0-r",
            "messages": [{"role": "user", "content": "Solve math problem"}],
            "stream": False,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "step 1: reasoning step 2: result"


def test_vllm_streaming_generation_mocked():
    serve_model.server_state.engine = "vllm"
    serve_model.server_state.backend = "vllm"

    class MockOutputText:
        def __init__(self, text: str):
            self.text = text

    class MockRequestOutput:
        def __init__(self, text: str):
            self.outputs = [MockOutputText(text)]

    class MockVLLMEngine:
        async def generate(self, prompt, sampling_params, request_id):
            yield MockRequestOutput("Hello ")
            yield MockRequestOutput("Hello world!")

    with TestClient(serve_model.app) as client:
        serve_model.server_state.vllm_engine = MockVLLMEngine()
        payload = {
            "model": "mau-llm-1.0-r",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "Hello" in response.text
