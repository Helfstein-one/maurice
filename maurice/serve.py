"""
maurice Inference Server (maurice/serve.py)
Implements an OpenAI-compatible /v1/chat/completions endpoint using FastAPI.
Supports Hugging Face Transformers, vLLM (AsyncLLMEngine), and Mock backends.
"""

import argparse
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "maurice"
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=512, ge=1)
    stream: bool | None = False


class ServerState:
    def __init__(self) -> None:
        self.variant: str = "c"
        self.backend: str = "hf"
        self.engine: str = "hf"
        self.model: Any = None
        self.tokenizer: Any = None
        self.vllm_engine: Any = None


server_state = ServerState()
app = FastAPI(title="maurice Inference Server")


def format_chatml_prompt(messages: list[ChatMessage], variant: str = "c") -> str:
    system_prompts = {
        "c": "You are maurice Code Agent (mau-llm-1.0-c), an expert AI coding and refactoring assistant.",
        "r": "You are maurice Reasoning Agent (mau-llm-1.0-r). Preserve and calibrate step-by-step chain-of-thought tokens inside <think>...</think> tags.",
        "g": "You are maurice General Agent (mau-llm-1.0-g), a helpful AI assistant.",
    }

    has_system = any(msg.role == "system" for msg in messages)
    prompt_parts = []

    if not has_system:
        sys_msg = system_prompts.get(variant, system_prompts["c"])
        prompt_parts.append(f"<|im_start|>system\n{sys_msg}<|im_end|>\n")

    for msg in messages:
        prompt_parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n")

    prompt_parts.append("<|im_start|>assistant\n")
    return "".join(prompt_parts)


@app.on_event("startup")
def load_model() -> None:
    engine_type = "hf"
    if server_state.engine in ["vllm", "mock"]:
        engine_type = server_state.engine
    elif server_state.backend in ["vllm", "mock"]:
        engine_type = server_state.backend

    server_state.engine = engine_type
    server_state.backend = engine_type

    if engine_type == "vllm":
        model_name_or_path = f"maurice-final-{server_state.variant}"
        logger.info(f"Initializing vLLM AsyncLLMEngine for model {model_name_or_path}...")
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            engine_args = AsyncEngineArgs(
                model=model_name_or_path,
                tensor_parallel_size=1,
                trust_remote_code=True,
            )
            server_state.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM AsyncLLMEngine initialized successfully.")
        except Exception as e:  # noqa: BLE001  # noqa: BLE001
            logger.warning(f"Could not initialize vLLM engine ({e}). Starting in fallback mode.")
            server_state.vllm_engine = None

    elif engine_type == "mock":
        logger.info("Server running in MOCK mode.")

    else:
        model_name_or_path = f"maurice-final-{server_state.variant}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            server_state.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            server_state.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            ).to(device)

            if hasattr(torch, "compile"):
                try:
                    server_state.model = torch.compile(server_state.model)
                    logger.info("torch.compile() applied successfully.")
                except Exception as e:  # noqa: BLE001  # noqa: BLE001
                    logger.warning(f"torch.compile() failed: {e}")

            logger.info(f"HF Model {model_name_or_path} loaded successfully.")
        except Exception as e:  # noqa: BLE001  # noqa: BLE001
            logger.warning(f"Could not load HF model ({e}). Starting in fallback mode.")


def _generate_hf_response(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    if server_state.model is None or server_state.tokenizer is None:
        return f"This is a mock response for mau-llm-1.0-{server_state.variant}."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = server_state.tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = server_state.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            top_p=top_p,
            pad_token_id=server_state.tokenizer.eos_token_id,
        )

    return server_state.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)


async def _generate_vllm_response(prompt: str, request: ChatCompletionRequest) -> str:
    if server_state.vllm_engine is None:
        return f"This is a mock response from vLLM engine for mau-llm-1.0-{server_state.variant}."

    temp = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 1.0
    max_tokens = request.max_tokens if request.max_tokens is not None else 512

    try:
        from vllm.sampling_params import SamplingParams

        sampling_params = SamplingParams(
            temperature=temp if temp > 0 else 0.0,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except ImportError:
        sampling_params = None

    request_id = f"vllm-{uuid.uuid4().hex}"
    results_generator = server_state.vllm_engine.generate(prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output and final_output.outputs:
        return final_output.outputs[0].text
    return ""


async def generate_vllm_stream(request: ChatCompletionRequest, prompt: str) -> AsyncGenerator[str, None]:
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if server_state.vllm_engine is None:
        mock_response = f"This is a mock streaming response from vLLM for mau-llm-1.0-{server_state.variant}."
        chunk_size = 10
        for i in range(0, len(mock_response), chunk_size):
            chunk = mock_response[i : i + chunk_size]
            response_obj = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(response_obj)}\n\n"
        final_obj = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_obj)}\n\n"
        yield "data: [DONE]\n\n"
        return

    temp = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 1.0
    max_tokens = request.max_tokens if request.max_tokens is not None else 512

    try:
        from vllm.sampling_params import SamplingParams

        sampling_params = SamplingParams(
            temperature=temp if temp > 0 else 0.0,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except ImportError:
        sampling_params = None

    request_id = f"vllm-{uuid.uuid4().hex}"
    results_generator = server_state.vllm_engine.generate(prompt, sampling_params, request_id)

    previous_text = ""
    async for request_output in results_generator:
        if request_output.outputs:
            text = request_output.outputs[0].text
            delta = text[len(previous_text) :]
            previous_text = text
            if delta:
                response_obj = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(response_obj)}\n\n"

    final_obj = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_obj)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_stream(request: ChatCompletionRequest, prompt: str) -> AsyncGenerator[str, None]:
    if server_state.engine == "vllm":
        async for chunk in generate_vllm_stream(request, prompt):
            yield chunk
        return

    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    full_response = _generate_hf_response(
        prompt,
        request.max_tokens or 512,
        request.temperature or 0.7,
        request.top_p or 1.0,
    )

    chunk_size = 10
    for i in range(0, len(full_response), chunk_size):
        chunk = full_response[i : i + chunk_size]

        response_obj = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(response_obj)}\n\n"

    final_obj = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_obj)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "variant": server_state.variant,
        "backend": server_state.backend,
        "engine": server_state.engine,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": f"mau-llm-1.0-{server_state.variant}",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "maurice",
            }
        ],
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> JSONResponse | StreamingResponse:
    prompt = format_chatml_prompt(request.messages, variant=server_state.variant)

    if request.stream:
        return StreamingResponse(generate_stream(request, prompt), media_type="text/event-stream")

    if server_state.engine == "vllm":
        response_text = await _generate_vllm_response(prompt, request)
    else:
        response_text = _generate_hf_response(
            prompt,
            request.max_tokens or 512,
            request.temperature or 0.7,
            request.top_p or 1.0,
        )

    return JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="maurice Inference Server")
    parser.add_argument("--variant", choices=["c", "r", "g"], default="c", help="Model variant")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--engine", choices=["hf", "vllm", "mock"], default="hf", help="Inference engine")
    parser.add_argument("--backend", choices=["hf", "vllm", "mock"], default=None, help="Backend alias for engine")
    return parser.parse_args(args_list)


def main(args_list: list[str] | None = None):
    args = parse_args(args_list)
    engine_val = args.backend if args.backend is not None else args.engine
    server_state.variant = args.variant
    server_state.engine = engine_val
    server_state.backend = engine_val

    logger.info(f"Starting server on {args.host}:{args.port} (variant: {args.variant}, engine: {engine_val})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
