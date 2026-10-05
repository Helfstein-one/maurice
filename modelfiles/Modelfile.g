# maurice Model File: mau-llm-1.0-g (General Purpose & Adaptive Reasoner)
# Optimized for zero-runtime C/C++ execution via llama.cpp and Ollama local deployment

FROM ../build/mau-llm-1.0-g-q4_k_m.gguf

# Model Hyperparameters & Temperature Limits
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# Stop Tokens for ChatML & Reasoning Blocks
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</think>"

# System Prompt Definition
SYSTEM """You are mau-llm-1.0-g, a general-purpose and adaptive reasoner in the maurice model suite. Balance instruction-following with calibrated thinking suppression for trivial conversational inputs."""

# Default ChatML Template Definition
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
