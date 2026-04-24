# AgentCF: Multimodal and Multi-agent for Recommendation

## Data process

You can download dataset from .......

```bash
python dataPrepare.py
```


## Agent Training

```bash
python AgentCF_train_check.py
```


## Agent Test

```bash
python AgentCF_Test_log-.py
```

## 代理配置
request.py是负责gpt系列的api, request1.py是负责glm系列的api(这里使用的模型是gpt-4o和glm-4.5)
注意修改使用不同模型的时候train文件和test文件的import都要修改看是request1还是request的python文件。
