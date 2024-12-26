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
# Install the package locally in editable model
pip install -e .
```

Edit the `.env` as needed:
```
cp example.env .env
```