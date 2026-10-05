FROM ../build/mau-llm-1.0-r-q4_k_m.gguf

# maurice Reasoning Model

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
SYSTEM """You are maurice-r (Reasoning variant). You are an advanced analytical assistant. You MUST ALWAYS use <think>...</think> tags to plan and reason step-by-step before providing your final answer."""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.6
PARAMETER top_p 0.95
