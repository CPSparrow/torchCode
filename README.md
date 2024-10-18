# 说明

- 目前还没有多少代码的修改，暂时不做说明。

# 论文（前置信息）阅读

- [ ] BPE和word-piece的区别
- [ ] GPT-2的full scoring和partial scoring指的是什么

# 计划

> - 阅读《大规模语言模型：从理论到实践》，重点可以关注章节**2、5、6、7**。自己查一些资料，搞清楚上下文学习和提示学习。每个章节，概念过完以后要跟着做代码实践。（11.17前完成）
> - 飞书上的要求：
    >

- 理解 GPT2\
  GPT2（单向Transformer）。\
  具体参照：
  [Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)

> - 理解 BERT\
    BERT的全称是Bidirectional Encoder Representation from
    Transformers（双向Transformer）。BERT在机器阅读理解顶级水平测试SQuAD1.1中表现出惊人的成绩:
    全部两个衡量指标上全面超越人类，并且在11种不同NLP测试中创出SOTA表现，包括将GLUE基准推高至80.4% (绝对改进7.6%)
    。BERT采用了Transformer的Encoder block 进行连接。\
    [知乎文章](https://zhuanlan.zhihu.com/p/103226488)
>   - 理解 RoBERTa\
      RoBERTa是BERT的改进版，通过改进训练任务和数据生成方式、训练更久、使用更大批次、使用更多数据等获得 了State of The
      Art的效果；可以用Bert直接加载。\
      GitHub[链接](https://github.com/brightmart/roberta_zh)
