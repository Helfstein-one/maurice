# maurice Model File: mau-llm-1.0-r (Pure Reasoning & Logic)
# Optimized for zero-runtime C/C++ execution via llama.cpp and Ollama local deployment

FROM ../build/mau-llm-1.0-r-q4_k_m.gguf

# Model Hyperparameters & Temperature Limits
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 50
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 4096

# Stop Tokens for ChatML & Reasoning Blocks
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# System Prompt Definition
SYSTEM """You are mau-llm-1.0-r, a pure reasoning engine in the maurice model suite. Preserve and calibrate step-by-step chain-of-thought tokens by placing your reasoning strictly inside <think>...</think> tags prior to presenting final solutions."""

# Default ChatML Template Definition
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
