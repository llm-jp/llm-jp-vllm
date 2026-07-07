#!/bin/bash
set -eoux pipefail

uv run ruff check llm_jp_vllm tests
uv run ruff format --check llm_jp_vllm tests
uv run mypy llm_jp_vllm
