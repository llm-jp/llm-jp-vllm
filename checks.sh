#!/bin/bash
set -eoux pipefail

uv run ruff check llm_jp_vllm
uv run ruff format --check llm_jp_vllm
uv run mypy llm_jp_vllm
