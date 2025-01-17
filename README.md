# LLM 报告 - **推理服务器**

## **概述**

该项目是一个简单的大型语言模型（LLM）推理服务器，用于处理提示并根据给定的参数生成响应。服务器读取一组提示，按批次进行处理，并将生成的文本输出到指定文件中。

## **项目结构**

```bash
.
├── decode.py
├── main.py
├── model
│   ├── config.json
│   ├── gpt2_pytorch_model.bin
│   └── vocab.json
├── prefill.py
├── prompts.py
├── requirements.txt
├── sampler.py
├── scheduler.py
└── test.txt
```

## **文件说明**

* `main.py`: 应用程序的入口点。它处理命令行参数并协调提示处理管道。
* `scheduler.py`: 管理提示批处理、编码、解码和生成响应。
* `prompts.py`: 包含服务器要处理的一组提示。
* `prefill.py`: 处理 prefill 过程，将 prompt 转换为 tokens, 计算 kv cache 并传输。
* `decode.py`: 处理 decode 过程，生成后续的新 token ID。
* `requirements.txt`: 列出运行项目所需的所有 Python 包。
* `sampler.py`: 实现不同的采样方法，如温度采样、Top-P 采样和 Top-K 采样。
* `model/`: 包含预训练模型文件 (`config.json`, `pytorch_model.bin`, `vocab.json`)。
* `test.txt`: 输出结果保存的文件。

## **环境配置**

1. 克隆仓库：
   ```bash
   git clone <https://github.com/fzw1202/llm-final-project.git>
   cd llm-final-project
   ```
2. 安装所需的依赖项：
   ```bash
   pip install -r requirements.txt
   ```
3. 下载 gpt2 模型 (手动)
   ```bash
   <https://huggingface.co/openai-community/gpt2/tree/main>
   ```

## **运行项目**

使用以下命令运行项目：

```bash
python main.py --batch-size <batch_size> --generation-length <length> --output-file <output_file>
```

### **命令行参数**

* `b, --batch-size`: 处理提示的批量大小（默认：4）。
* `l, --generation-length`: 生成内容的长度（默认：100）。
* `o, --output-file`: 输出文件的名称（默认：`test.txt`）。

## **示例用法**

```bash
python main.py -b 4 -l 100 -o test.txt
```

这将按批量大小为 4 处理 prompt，每个提示生成最多 100 个 token，并将结果保存到 `test.txt`。

## Prompt

`prompts.py` 文件包含服务器将处理的一些示例 prompt。可以根据需要修改此列表或添加新 prompt。

```python
prompts = [
    "In the future, artificial intelligence will",
    "Scientists have discovered a new way to",
    "On a distant planet, aliens are",
    "One of the greatest inventions in human history is",
    "In a peaceful village, the residents daily",
    "In a mysterious forest, explorers found",
    "In a bustling city, people are",
    "In an ancient castle, there is a hidden",
]
```

## Sampler

`sampler.py` 文件实现了不同的采样方法，包括温度采样、Top-P 采样和 Top-K 采样。

### **温度采样**

通过调整logits的温度值来控制生成文本的随机性。温度值越低，生成的文本越确定；温度值越高，生成的文本越多样化。

### **Top-P采样**

通过对概率分布进行累积求和，选择累积概率超过设定阈值的token进行采样。

### **Top-K采样**

选择概率最高的前K个token进行采样。


## **注意事项**

* 确保 `scheduler.py`中指定的GPU设备（`cuda:0` 和 `cuda:1`）在您的系统上可用。
