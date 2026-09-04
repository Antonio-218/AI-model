# DeepSeek-R1-Distill-Qwen-7B LoRA 微调流程（Unsloth）

> 面向 ERC20 交易问答场景的监督微调（SFT）完整流程，运行环境为 Jupyter Notebook / Colab + 单卡 GPU。

---

## 1. 安装依赖

```python
!pip install datasets
!pip install peft accelerate bitsandbytes
!pip install unsloth
```

| 依赖 | 作用 |
| --- | --- |
| `unsloth` | 用于加速大语言模型微调的库 |
| `datasets` | Hugging Face 提供的数据集库 |
| `peft` | 高效参数微调库（Parameter-Efficient Fine-Tuning） |
| `accelerate` | 分布式训练库 |
| `bitsandbytes` | 实现 4bit / 8bit 量化训练的库 |

---

## 2. 登录 Hugging Face

建立与 Hugging Face 模型中心的连接。

```python
from huggingface_hub import login

login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxx")  # 替换为你的实际 token
```

---

## 3. 加载模型与分词器

```python
from unsloth import FastLanguageModel      # 导入 unsloth 中的 FastLanguageModel 类
from transformers import AutoTokenizer     # 导入 transformers 库中的 AutoTokenizer 类
import torch                               # 导入 PyTorch 库

# 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",  # 指定模型名称
    max_seq_length=2048,      # 设置模型的最大序列长度
    dtype=torch.bfloat16,     # 使用 bfloat16 数据类型以优化计算效率
    load_in_4bit=True         # 以 4 位量化方式加载模型以节省显存
)
```

---

## 4. 注入 LoRA 模块

为模型注入 LoRA 模块，实现高效微调。

```python
# 打印加载的模型和分词器信息
print(f"Model: {model}")
print(f"Tokenizer: {tokenizer}")

from peft import LoraConfig, get_peft_model

# 配置并注入 LoRA 模块
model = FastLanguageModel.get_peft_model(
    model,                    # 待修改的模型
    r=64,                     # LoRA 的秩（r），控制参数数量
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],                        # 需要注入 LoRA 的目标模块
    lora_alpha=16,            # LoRA 的缩放比例
    lora_dropout=0.05,        # LoRA 模块的 dropout 概率
    bias="none",              # 不使用偏置项
    use_gradient_checkpointing=True   # 启用梯度检查点以节省显存
)
```

---

## 5. 微调前验证生成效果

```python
prompt = "交易哈希为 0x1630239127e29a29c51f0855c5ef1100775901621e9ff483b9c97f5ca3c44e0b 的交易用了多少 gas？"

# 将提示文本转化为模型输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 使用模型生成文本
outputs = model.generate(**inputs, max_new_tokens=300)

# 解码并打印生成结果
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 6. 加载数据集

```python
from datasets import load_dataset

# 从 JSON 文件加载数据集，并指定为训练集
dataset = load_dataset(
    "json",
    data_files={"train": "erc20_chat_train_samples.json"},
    split="train"
)

# 划分训练集 / 测试集，测试集占 10%
dataset = dataset.train_test_split(test_size=0.1)

print(dataset["train"][0])   # 打印训练集中的第 1 个样本
```

---

## 7. 定义 `formatting_func`（转换成训练样本）

遍历 `messages` 列表，根据 `role` 字段把内容分别拼接成「用户问题」与「机器人回答」。

```python
def formatting_func(example):
    messages = example["messages"]
    user_message = ""
    assistant_message = ""      # 初始化两个空字符串，用于存储用户和助手的消息

    for message in messages:
        if message["role"] == "user":
            user_message += message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            assistant_message += message["content"].strip() + "\n"

    # strip() 移除内容两端空白，\n 用于在消息之间添加换行
    formatted = f"""### 用户问题：
{user_message.strip()}

### 机器人回答：
{assistant_message.strip()}"""

    return [formatted]          # 必须返回 list

print(formatting_func(dataset["train"][2])[0])
```

---

## 8. 格式转换

把格式化后的文本写入 `text` 字段。

```python
def apply_format(example):
    return {"text": formatting_func(example)[0]}

dataset = dataset.map(apply_format)

print(dataset["train"][0])      # 打印训练集的第 1 条样本
```

---

## 9. 设置训练参数

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs",                    # 输出目录，保存模型与日志
    per_device_train_batch_size=2,           # 每个设备的训练批次大小
    per_device_eval_batch_size=2,            # 每个设备的评估批次大小
    gradient_accumulation_steps=2,           # 梯度累积步数，模拟更大 batch
    warmup_steps=10,                         # 预热步数，逐步调整学习率
    max_steps=50,                            # 最大训练步数
    learning_rate=2e-4,                      # 学习率
    logging_steps=1,                         # 每隔多少步记录一次日志
    fp16=True,                               # 启用混合精度训练
    save_steps=25,                           # 保存模型的步数间隔
    eval_strategy="steps",                   # 评估策略：按步数评估
    eval_steps=10,                           # 评估步数间隔
    save_total_limit=1,                      # 保存模型的总数量上限
    report_to="none"                         # 不使用任何报告工具
)
```

---

## 10. 训练器初始化

```python
from unsloth.trainer import SFTTrainer

trainer = SFTTrainer(
    model=model,                             # 要训练的模型
    tokenizer=tokenizer,                     # 用于预处理和编码的 tokenizer
    train_dataset=dataset["train"],          # 训练数据集
    eval_dataset=dataset["test"],            # 评估数据集
    dataset_text_field="input",              # 数据集中包含文本的字段名
    max_seq_length=2048,                     # 序列最大长度，超出截断
    args=training_args                       # 训练参数
)
```

---

## 11. 模型训练

```python
trainer.train()
```

---

## 12. 微调后验证

```python
prompt = "### 用户问题:\n交易哈希为 0x1630239127e29a29c51f0855c5ef1100775901621e9ff483b9c97f5ca3c44e0b 的交易用了多少 gas?"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=300)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 13. 保存模型

```python
# 保存训练好的模型
trainer.model.save_pretrained("./fine_tuned_erc20_model")
# 保存分词器，确保加载模型时使用相同的文本处理方式
tokenizer.save_pretrained("./fine_tuned_erc20_model")
```

---

## ⚠️ 代码中的几处问题与修正建议

### 1. `dataset_text_field` 字段名写错（会导致训练报错）

格式化后写入的字段是 `text`，但训练器里填的是 `"input"`，两处必须一致：

```python
dataset_text_field = "text"   # 原来是 "input"
```

### 2. `fp16=True` 与 4bit 量化模型的搭配

`bitsandbytes` 的 4bit 模型在部分版本下与 `fp16` 混用会出现数值不稳定（`bfloat16` / `fp16` 混用告警）。若卡在 A100 / H100 / 4090 等支持 bf16 的卡上，建议：

```python
bf16=True,
fp16=False,
```

### 3. `eval_strategy` 的参数名

旧版 `transformers` 中该参数名为 `evaluation_strategy`。若报 `TypeError: unexpected keyword argument 'eval_strategy'`，改名即可。

### 4. 生成阶段建议加上 `<｜begin▁of▁sentence｜>`

DeepSeek-R1 系列蒸馏模型依赖 BOS token，微调后验证时建议：

```python
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
```

并确认 `tokenizer.pad_token` 已设置（Unsloth 通常会自动处理，若报缺失则 `tokenizer.pad_token = tokenizer.eos_token`）。

### 5. 注释与代码不一致的小瑕疵

- `print(dataset["train"][0])` 的注释原文写的是"打印第 4 个样本"，实际是第 1 个。
- `formatting_func` 中「遍历 messages 列表」等注释写在 f-string 内部，建议移到函数体外，避免污染训练文本。

### 6. Hugging Face Token 安全

原代码中直接明文写入了 token。若该 token 已经提交到公开仓库或分享过，请立即到 Hugging Face → Settings → Access Tokens 中 **revoke 并重新生成**，改用环境变量读取：

```python
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])
```
