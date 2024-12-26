# Polis Argument Mapping to Guide Policy Decisions

## Research Objective

**How can large language models enable us to ingest massive streams of unstructured information, incorporate diverse perspectives and distill them into actionable insights that demonstrably align with public opinion?**

## Installation

Assuming you use mamba -- can be remplaced by conda.

```
# Set the environment
mamba env create -n argmap -f conda-argmap.yml
# Activate the environment
mamba activate argmap
```

Copy the example `.env` and modify as needed:
```
cp example.env .env
```

The library should run on CPU if no CUDA support is detected. Some of the notebooks still have the old requirement for GPUs, as illustration, but that might be adapted in the future.

It also supports the latest model `modernBERT` from answerdotai: just specificy 
```
EMBED_MODEL_ID = answerdotai/ModernBERT-base
```
in your `.env` file.
