# Seeing and Reflecting: Multimodal Memory-Enhanced Agent Collaboration for Recommendation

## Paper URL: https://arxiv.org/abs/2607.07108

We introduce MMEACR, a Multimodal Memory-Enhanced Agent Collaboration for Recommendation.
<div align="center">
  <img src="doc/MMEACR.png" alt="Logo" style="width:100%;">
</div>

MMEACR achieves great improvement in CDs, Cell Phones and Fashion.
<div align="center">
  <img src="doc/doc2.png" alt="Logo" style="width:100%;">
</div>


## Data process

You can download dataset from .......

```bash
python dataPrepare.py
```


## Agent Training

```bash
python AgentCF_train_check.py
```


## Agent Testing

```bash
python AgentCF_Test_log-.py
```

## Proxy Configuration
request.py handles the API for the GPT series, while request1.py is responsible for the GLM series (the models used here are gpt-4o and glm-4.5). When switching between different model series, make sure to update the import statements in both the train and test files to reference either request or request1 accordingly. Additionally, modify the config.py file to set model = "glm-4.5" or model = "gpt-4o" as needed.


