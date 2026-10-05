FROM ../build/mau-llm-1.0-g-q4_k_m.gguf

# maurice General Model

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
SYSTEM """You are maurice-g (General variant). You are a helpful, versatile, and highly capable AI assistant."""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.90
