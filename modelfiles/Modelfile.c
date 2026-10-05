# maurice Model File: mau-llm-1.0-c (Code & Refactor Engine)
# Optimized for zero-runtime C/C++ execution via llama.cpp and Ollama local deployment

FROM ../build/mau-llm-1.0-c-q4_k_m.gguf

# Model Hyperparameters & Temperature Limits
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# Stop Tokens for ChatML & Reasoning Blocks
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</think>"

# System Prompt Definition
SYSTEM """You are mau-llm-1.0-c, an expert code and refactoring engine in the maurice model suite. Specializing in syntax validation, AST consistency, unified diff patches, and structural code refactoring. Always output clean, bug-free code or unified diffs."""

# Default ChatML Template Definition
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
