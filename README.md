# 说明

对于transformers等库相关的代码，可以在对应目录使用以下命令一键启动：

```
export HF_ENDPOINT=https://hf-mirror.com && python main.py
```

### 准备测试的几个模型：

```
Langboat/bloom-1b4-zh //效果一般
IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese
```

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
