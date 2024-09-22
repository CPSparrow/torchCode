# 知识点

- [ ] 查看```nn.model```的文档
- [ ] ```optim.SGD(..., momentum=0.99)```后一个参数怎么理解
- [ ] ```dataloader```的参数的含义
- [ ] ```LongTensor```的作用
- [ ] ```position_encoding```的```forward```
- [ ] 根据这里的代码中的dec_inputs和dec_outputs,是否就是资料中所提到的teacher forcing?
- [ ] ```model(enc_inputs, dec_inputs)```为什么可以不调用```forward```?
- [ ] 在这个代码示例里面，word embedding是不包含语义信息的吗？
- [ ] torch中的DataSet库
- [ ] ```Linear```设置为```bias=False```对训练会有什么影响？
- [ ] MHA的residual为何使用的是input_Q?有空应当完整的检查一下代码看看是否有区别。

# 运行

- [ ] 多次运行该代码会改变结果吗？(输出的随机性)
- [ ] ```register_buffer()```
- [ ] 查看```make_data()```的返回值的形状
- [ ] 尝试修改代码，计算训练时的perplexity
- [ ] ```MyDataSet```相关的代码可以优化
- [ ] 自己实现一个tokenizer?

# 笔记

- [ ] 这里有一个实现的技巧在于：src和tgt vocab都把padding对应了0,从而与其中对应的代码结合。而与之相对应的，开始标记S与中止标记E的索引则无关紧要。
