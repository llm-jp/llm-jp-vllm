# vLLM plugins for LLM-jp models

This repository contains vLLM plugins which are necessary to launch LLM-jp models
with vLLM entrypoints.

## Installation

```shell
uv pip install llm-jp-vllm
```

The above command installs `llm-jp-vllm` directory onto your environment's `site-packages`.

## Usage

This repository works with the plugin mechanism on vLLM.

After [necessary change in vLLM](https://github.com/vllm-project/vllm/pull/45241) was merged,
you can use reasoning/tool parsers implemented in this repository
by passing appropriate module names to `--reasoning/tool-parser-plugin` option.

```shell
vllm serve {llm-jp-4 model} \
  --trust-remote-code \
  --reasoning-parser llmjp4 \
  --reasoning-parser-plugin llm_jp_vllm.llmjp4
```

### Tool calling

The `llmjp4` tool parser converts Harmony-format tool calls (including
the parallel call extension of LLM-jp-4) into OpenAI-compatible
`tool_calls`, for both streaming and non-streaming responses:

```shell
vllm serve {llm-jp-4 model} \
  --trust-remote-code \
  --reasoning-parser llmjp4 \
  --reasoning-parser-plugin llm_jp_vllm.llmjp4 \
  --enable-auto-tool-choice \
  --tool-call-parser llmjp4 \
  --tool-parser-plugin llm_jp_vllm.llmjp4
```

## Development

```shell
uv sync --all-extras --dev
./checks.sh  # ruff / mypy
uv run pytest
```