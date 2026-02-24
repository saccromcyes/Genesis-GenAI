# Download Model Instructions (No API Keys)

GENESIS requires a GGUF model for llama.cpp.

Steps:
1) Download a GGUF instruct model from a trusted source (e.g., HuggingFace).
2) Put it into:
   models/llm.gguf
3) Run:
   python -m genesis.cli --model models/llm.gguf --ingest sample_kb --question "Hello GENESIS"
