# My-Transformer

一个使用 PyTorch 从零实现 Transformer 的中英翻译练习项目。

项目包含 Transformer Encoder、Decoder、多头注意力、位置编码、mask、Noam 学习率调度、训练循环和 greedy decoding，并使用 `zh.txt` / `en.txt` 中的简单平行语料做演示训练。

## 项目结构

```text
.
├── transformer.py
├── zh.txt
└── en.txt
```

## 主要功能

- 从零实现 Transformer 基础模块
  - Positional Encoding
  - Multi-Head Attention
  - Feed Forward Network
  - Encoder Layer / Decoder Layer
  - Encoder / Decoder / Transformer
- 支持 padding mask 和 decoder subsequent mask
- 使用 Noam 学习率调度策略
- 自定义中英平行语料 Dataset 和 DataLoader
- 支持训练、验证和 greedy decoding 推理

## 环境要求

- Python 3.8+
- PyTorch
- NumPy
- tqdm

安装依赖示例：

```bash
pip install torch numpy tqdm
```

如果需要 GPU 训练，请按自己的 CUDA 版本安装对应的 PyTorch 版本。

## 运行

在仓库根目录执行：

```bash
python transformer.py
```

脚本会读取：

- `zh.txt`：中文句子
- `en.txt`：对应英文句子

两份文件需要按行一一对应。

## 默认训练配置

脚本中的默认配置适合小数据集演示：

```text
d_model: 128
n_layers: 2
n_heads: 4
d_ff: 256
max_len: 64
batch_size: 8
epochs: 80
```

训练完成后，脚本会用 greedy decoding 对几条中文测试句子进行翻译。

## 说明

这个项目更偏向 Transformer 原理学习和代码拆解，不是面向大规模机器翻译任务的完整训练框架。当前语料规模较小，输出质量主要用于验证模型流程是否跑通。

可以继续改进的方向：

- 扩充平行语料
- 增加 tokenizer / BPE / SentencePiece
- 增加 checkpoint 保存与加载
- 增加 BLEU 等评估指标
- 拆分模型、数据、训练和推理模块
- 支持 beam search decoding
