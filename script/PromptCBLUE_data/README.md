# PromptCBLUE_data

| 任务名        | 状态       |
| ------------ | ---------------------------------------|
| CMeEE-V2     |  ✅        |
| CMeIE        |  ✅        |
| CHIP-CDN     |  ✅        |
| CHIP-CDEE    |  ✅        |
| IMCS-V2-NER  |  ✅        |
| CHIP-MDCFNPC |  对话       |
| IMCS-V2-SR   |  对话      |
| IMCS-V2-DAC  |  对话       |
| CHIP-CTC     |  ✅        |
| CHIP-STS     |  ✅        |
| KUAKE-IR     |  原始数据太大      |
| KUAKE-QIC    |  ✅        |
| KUAKE-QQR    |  暂时不理解如何从3分类变成4分类的|
| KUAKE-QTR    |  ✅        |
| MedDG        |  ✅        |
| IMCS-V2-MRG  |  对话      |

## 构造数据集

### CBLUE原始训练集
下载 [中文医疗信息处理评测基准CBLUE](https://tianchi.aliyun.com/dataset/95414)
解压到同一文件夹下：
```
(bgillm) root@yourui-gpu-dev:CBLUE# tree
.
├── CHIP-CDEE
│   ├── CHIP-CDEE_dev.json
│   ├── CHIP-CDEE_test.json
│   ├── CHIP-CDEE_train.json
│   ├── README.txt
│   ├── example_gold.json
│   ├── example_pred.json
│   └── 国际疾病分类ICD-10北京临床版v601.xlsx
├── CHIP-CDN
│   ├── CHIP-CDN_dev.json
│   ├── CHIP-CDN_test.json
│   ├── CHIP-CDN_train.json
│   ├── README.txt
│   ├── example_gold.json
│   ├── example_pred.json
│   ├── 国际疾病分类 ICD-10北京临床版v601.xlsx
│   └── 手术标注词2500.tsv
├── CHIP-CTC
│   ├── CHIP-CTC_dev.json
│   ├── CHIP-CTC_test.json
│   ├── CHIP-CTC_train.json
│   ├── README.txt
│   ├── category.xlsx
│   ├── example_gold.json
│   └── example_pred.json
......
├── MedDG
│   ├── MedDG_dev.json
│   ├── MedDG_test.json
│   ├── MedDG_train.json
│   ├── README.txt
│   ├── entity_list.txt
│   ├── evaluate.py
│   ├── example_gold.json
│   └── example_pred.json
└── Text2DT
    ├── Text2DT_dev.json
    ├── Text2DT_eval.zip
    ├── Text2DT_test.json
    ├── Text2DT_train.json
    └── example_pred.json
```
### PromptCBLUE训练集
下载PromptCBLUE训练数据(通过训练数据猜测prompt模式)  
[CCKS2023-PromptCBLUE中文医疗大模型评测基准—开源赛道](https://tianchi.aliyun.com/competition/entrance/532084/introduction)

### 构造训练集

```
usage: PromptCBLUE-data [-h] --cblue CBLUE --promptcblue PROMPTCBLUE --temp TEMP --out OUT

Build PromptCBLUE train data

optional arguments:
  -h, --help                show this help message and exit
  --cblue CBLUE             CBLUE data directory            # CBLUE训练集目录
  --promptcblue PROMPTCBLUE promptcblue training data file  # PromptCBLUE训练集的train.json文件
  --temp TEMP               temp directory                  # 临时文件目录
  --out OUT                 output file directory           # 输出文件目录
```

```shell
python PromptCBLUE-data.py \
    --cblue CBLUE \
    --promptcblue train.json \
    --temp data \
    --out data/BGI
```
实际训练文件`sha256`:
```
data/BGI/CHIP-CDEE_bgi_train_1587.json	421bf7a2f614e254d7b7be705fe6ed320ecbb4266d72a847328a2c92c6f58a63
data/BGI/CHIP-CDN_bgi_train_6000.json	e984ecd24efa13374ee4ca481bf31a5a7a648b425e900e7e3af2b86003769584
data/BGI/CHIP-CTC_bgi_train_22962.json	b19ad3054760f63b17e27fffc3fd6d36441bac6581ddbe5e943844649da1f972
data/BGI/CHIP-STS_bgi_train_16000.json	7fd8e6d9ba5961c54330a557e15984fe34d91d4f45661b89f71652c9001f8786
data/BGI/CMeEE_bgi_train_15000.json	70eacfad90b6a91b524f45f84359a4106c58ea318ec1bde3e78312d5f1a41878
data/BGI/CMeIE_bgi_train_14291.json	77f9401bda6c8a1fca2964c292e4b02ae587f9bfd5981b02c1815f11830ba492
data/BGI/IMCS-V2-NER_bgi_train_41765.json	b599f39922f593c38026af0b0a653e3c2b8f97ee0292200002f2c1d3e5727e38
data/BGI/KUAKE-IR_bgi_train_10000.json	30286cceb5cc3d14b208b3af668de7ea61649c143b7c794982997519cb4411aa
data/BGI/KUAKE-QIC_bgi_train_5314.json	7009b9dea7d5f0d17f4c46cffefa6a993a5674a010d15328aeb2ee4ae8034d88
data/BGI/KUAKE-QTR_bgi_train_24174.json	e95bce9686c88158ab412d829b826021590afa9d192218f6c10819cff52890ec
data/BGI/MedDG_bgi_train_10000.json	db1027595da57192cd4f11db434a716b241ad88a7c15f43aec0d97ba92a72d8b
data/BGI/train.json	b0f5aa975f57ed43eafe85bc37834fa3e0be648b0ca14082553ab5aa9022a8a9
```

最终训练集包括：构造的数据集和PromptCBLUE训练集的`train.json`文件