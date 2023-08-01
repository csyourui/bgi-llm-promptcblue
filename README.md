# bgi-llm-promptcblue

## Training procedure
我们在baichuan-13b模型基座上，对全量参数进行了有监督的微调, (区别于通用的sft方案，我们在计算loss按照预训练的策略对于全部的token均计算了loss)

[https://huggingface.co/yourui/bgi-promptcblue-baichuan-13b](https://huggingface.co/yourui/bgi-promptcblue-baichuan-13b)
> 选择了`step=50000`的checkpoints作为最终的模型(max_steps=58920)

微调：
```shell
chmod 755 ./promptcblue/supervised_finetuning/fintune.sh
./promptcblue/supervised_finetuning/fintune.sh
```
### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 2e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- total_train_batch_size: 8
- total_eval_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 2.0

### Framework versions

- Transformers 4.30.2
- Pytorch 2.0.1+cu118
- Datasets 2.12.0
- Tokenizers 0.13.3

## Data
在**PromptCBLUE**的基础训练数据上，扩充了到了235k条训练数据, 具体扩充方法见[PromptCBLUE_data](script/PromptCBLUE_data/README.md)
训练数据文件：[file](data/PromptCBLUE.zip)

数据: total: 166779
| type         | 训练数据    |
| ------------ | -----------|
| train.json   |  68900     |
| CMeEE-V2     |  15000     |
| CMeIE        |  14291     |
| CHIP-CDN     |  6000      |
| CHIP-CDEE    |  1587      |
| IMCS-V2-NER  |  41765     |
| CHIP-MDCFNPC |  0         |
| IMCS-V2-SR   |  0         |
| IMCS-V2-DAC  |  0         |
| CHIP-CTC     |  22962     |
| CHIP-STS     |  16000     |
| KUAKE-IR     |  10000     |
| KUAKE-QIC    |  5000      |
| KUAKE-QQR    |  0         |
| KUAKE-QTR    |  24174     |
| MedDG        |  10000     |
| IMCS-V2-MRG  |  0         |

prompt处理如下：

```python
f"Write a response that appropriately completes the Input.\n\nInput:\n{input}\n\nResponse:\n{target}{LLAMA_EOS_TOKEN}"
```

## Generate
下载模型[https://huggingface.co/yourui/bgi-promptcblue-baichuan-13b](https://huggingface.co/yourui/bgi-promptcblue-baichuan-13b)，并保存在model目录下

> 为了加速推理，推理数据分成八份，每份由一张卡推理。

```shell
chmod 755 ./script/PromptCBLUE_generate/generate_all.sh 
chmod 755 ./script/PromptCBLUE_generate/baichuan/generate.sh

./script/PromptCBLUE_generate/generate_all.sh baichuan
```