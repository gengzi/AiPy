## 分词过程



分词目的：将人类语言“翻译”或者“拆解”成 AI 能听懂的、最小的“零件”。
AI 是个“数字大脑”它只认识数字，所以将人类语言转换成数字，才能让 AI Understand。
tokenizers 流程：文字 → 拆成小块 → 小块变数字 → 送给 AI 处理
### 分词策略
1. 按字拆（常见于中文） 
2. 按词拆（更聪明一点）
```angular2html
"我"     -> token ID: 100
"爱"     -> token ID: 200
"吃"     -> token ID: 300
"苹果"   -> token ID: 700  （因为“苹果”是一个词） 这样拆更高效，因为“苹果”两个字被当成一个整体，省了一个 token
"。"     -> token ID: 600
```
### 词表
每一个 AI 模型在“出生”（训练）的时候，就自带一本专属的“词典”——这本词典就是它的“词表”（Vocabulary），里面清清楚楚地写着：哪个词（或字、子词）对应哪个数字 ID
例如：
```angular2html
Token（文字块）	对应的数字 ID
吃	100
苹果	200
。	500
[UNK]	0         
[PAD]	1
```
那如果遇到没见过的词呢？
如果假如词表中没有“犇”也不在词表里 → 标记为 [UNK]（unknown），那解析后的数字可能就是 0
#### 词表文件：
当你下载一个 Hugging Face 模型（比如用 from_pretrained("bert-base-chinese")），它会在本地生成几个文件，其中最关键的词表文件通常是：
```angular2html
文件名	         说明
vocab.txt	最常见！ 用于 BERT、RoBERTa 等模型，每行一个 token
tokenizer.json	更现代的格式，包含词表 + 分词规则（由 tokenizers 库生成）
merges.txt	子词合并规则，配合 vocab.json 使用（如 GPT-2、BPE 算法）
vocab.json	token 到 ID 的映射，JSON 格式
special_tokens_map.json	特殊标记，如 [CLS]、[SEP]、[PAD] 等
tokenizer_config.json	分词器的配置，比如用的是 WordPiece 还是 BPE
```
1. vocab.txt（BERT 中文模型片段）
行号 = ID，内容 = token   所以 "我" 的 ID 是 5（从 0 开始数），"苹果" 的 ID 是 10
```angular2html
[UNK]
[S]
]
##
我
爱
吃
苹
果
苹果
今天
天气
真
好
...
```
2. vocab.json（JSON 格式，如 GPT-2）
```angular2html
{
  "我": 100,
  "爱": 200,
  "吃": 300,
  "苹果": 400,
  "今天": 500,
  "[UNK]": 0,
  "[PAD]": 1,
  ...
}
```
3. tokenizer.json（全能型文件）
二进制+JSON混合的高效格式.包含
```angular2html
词表（vocab）
合并规则（merges）
特殊 token
分词算法（BPE、WordPiece 等）
语言配置
```
#### 模型是如何使用词表的
* 你调用 tokenizer("我爱吃苹果")
* tokenizer 去读本地的 vocab.txt 或 tokenizer.json
* 查表得到：["我", "爱", "吃", "苹果"] → [100, 200, 300, 400]
* 把这个数字序列喂给模型
* 模型内部的“嵌入层”（Embedding）会根据这些 ID，找到对应的向量表示
```angular2html
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 查看词表大小
print("词表大小:", tokenizer.vocab_size)  # 比如 21128

# 查某个 token 的 ID
print(tokenizer.convert_tokens_to_ids(["我", "爱", "吃", "苹果"]))  # [100, 200, 300, 400]

# 查某个 ID 对应什么
print(tokenizer.convert_ids_to_tokens([100, 200, 300, 400]))  # ['我', '爱', '吃', '苹果']

# 查特殊 token
print(tokenizer.cls_token_id)  # [CLS] 的 ID
print(tokenizer.unk_token_id)  # [UNK] 的 ID
```


### Hugging Face Tokenizers 库
tokenizers 是 Hugging Face 开发的一个高性能分词库，它的作用是：
* 帮你快速加载模型对应的分词器
* 高效地把文本转成 token ID
支持 BPE、WordPiece、Unigram 等各种智能分词算法
但它不会“自己决定”词表——它是严格按照模型的词表来工作的。
```python
from transformers import AutoTokenizer, AutoModel

# ✅ 一定要从同一个模型名加载！
model_name = "uer/roberta-base-chinese"  # 中文模型

tokenizer = AutoTokenizer.from_pretrained(model_name)  # 配套的 tokenizer
model = AutoModel.from_pretrained(model_name)           # 对应的模型

# 现在分词，模型一定认识！
text = "今天天气真好"
inputs = tokenizer(text, return_tensors="pt")  # 输出: input_ids, attention_mask 等

# 查看 tokenizer 的词汇表大小
print(tokenizer.vocab_size)  # 比如 21128

# 查看某个词的 ID
print(tokenizer.convert_tokens_to_ids(["我", "爱", "吃"]))  # [100, 200, 300]

# 查看 ID 对应什么词
print(tokenizer.convert_ids_to_tokens([100, 200, 300]))  # ['我', '爱', '吃']
```
### Token
tokenizers 拆分后的数字，就是一个token
中文一般一个字 ≈ 1 个 token（但也不绝对，看模型）
英文一个词可能拆成多个 token
Token 数量 = 大模型花费/速度的关键！