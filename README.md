# ValueDCG
Code repository for ValueDCG: Measuring Comprehensive Human Value Understanding Ability of Language Models

## Quick Start

You should first add an `openai_key_info.py` file in the root file, with openai key and url inside.

Then you should set the PYTHONFILE to ValueDCG file.

Finally you can perform the experiments, namely:

```
git clone https://github.com/zowiezhang/ValueDCG.git
cd ValueDCG
export PYTHONFILE=$(pwd)
pip install -r requirements.txt
python get_response.py
```

## Experiment Settings

Our LLM parameter settings are as following:

```
temperture = 0.0,
top_p = 0.95,
seed = 42
```

And we use `get-4o-2024-05-13` version as the evaluator.

## Repo structure

Consistency Experiments in `experiments`

Full experiment record in `response`

