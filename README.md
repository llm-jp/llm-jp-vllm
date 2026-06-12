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