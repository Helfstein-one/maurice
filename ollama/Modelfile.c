FROM ../build/mau-llm-1.0-c-q4_k_m.gguf

# maurice Code Model
# Based on DeepSeek-R1-Distill-Qwen-1.5B (simulated)

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
SYSTEM """You are maurice-c (Coding variant). You are an ultra-fast programming assistant. You generate clean, optimal, and thoroughly documented code."""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
