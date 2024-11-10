# 说明

对于transformers等库相关的代码，可以在对应目录使用以下命令一键启动：

```
export HF_ENDPOINT=https://hf-mirror.com && python BERT_train.py
```

数据集下载命令：

```
export HF_ENDPOINT=https://hf-mirror.com && huggingface-cli download --repo-type dataset wikipedia --local-dir wikipedia
```

### 接下来要看的论文

- Improving Language Model Reasoning with Self-motivated
- RAT
- Can Small Language Models Help Large Language Models
- Automatic Task-Level Thinking Steps Help Large Language Model..
- G-EVAL
- Xot
  ![论文图片](/doc_img/论文.png)

### 论文（前置信息）阅读

- [ ] BPE和word-piece的区别。
- [ ] GPT-2的full scoring和partial scoring指的是什么。
- [ ] 各种优化器的发展历程和比较。

### 计划

> - 《大规模语言模型：从理论到实践》重点章节：**2、5、6、7**。每个章节，概念过完以后要跟着做代码实践。（11.17前完成）
> - GPT2:
    [Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)
> - BERT:[知乎文章](https://zhuanlan.zhihu.com/p/103226488)
> - RoBERTa:[链接](https://github.com/brightmart/roberta_zh)
