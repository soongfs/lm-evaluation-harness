# KQA-Pro 任务

这里存放仓库内本地使用的 KQA-Pro 任务定义。

目录结构：

- `kqa_pro_mcq/`：基础版多项选择任务
- `kqa_pro_mcq_instruct/`：面向 instruct 模型的生成式多项选择任务
- `kqa_pro_qa/`：自由生成式 QA 任务

使用方式：

```bash
--include_path ./kqa_pro_tasks
```

每个任务目录都是自包含的，并从 `dataset/train.json` 和 `dataset/val.json` 加载本地数据。

其中 MCQ 任务分成两条与 MMLU 风格对应的评测线：

- `kqa_pro_mcq`：continuation 风格的多项选择评测，比较 `A-J` 选项标签
- `kqa_pro_mcq_instruct`：instruct/generative 风格的多项选择评测，并使用严格答案抽取
