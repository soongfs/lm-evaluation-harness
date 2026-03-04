# KQA-Pro 任务

这里存放仓库内本地使用的 KQA-Pro 任务定义。

目录结构：

- `kqa_pro_mcq/`：基础版多项选择任务
- `kqa_pro_mcq_instruct/`：面向 instruct 模型的 continuation 风格多项选择任务
- `kqa_pro_qa/`：面向基础模型的自由生成式 QA 任务
- `kqa_pro_qa_instruct/`：面向 instruct 模型的自由生成式 QA 任务
- `kqa_pro_qa_contains/`：面向基础模型的 contains 指标 QA 任务
- `kqa_pro_qa_contains_instruct/`：面向 instruct 模型的 contains 指标 QA 任务

使用方式：

```bash
--include_path ./kqa_pro_tasks
```

每个任务目录都是自包含的，并从 `dataset/train.json` 和 `dataset/val.json` 加载本地数据。

其中 MCQ 任务分成两条与 MMLU 风格对应的评测线：

- `kqa_pro_mcq`：continuation 风格的多项选择评测，比较 `A-J` 选项标签
- `kqa_pro_mcq_instruct`：参考 Llama 3 instruct MMLU continuation 模板的多项选择评测，生成短答案并直接做 `exact_match`

QA 任务目前分成两条与 `nq_open` 风格接近的评测线：

- `kqa_pro_qa`：基础模型用的开放式 QA，直接生成短答案并用内置 `exact_match` 评测
- `kqa_pro_qa_instruct`：instruct 模型用的开放式 QA，同样直接生成短答案并用内置 `exact_match` 评测
- `kqa_pro_qa_contains`：基础模型用的宽松 QA 评测，采用 CoT 风格输出，只要归一化后的答案文本在回复中出现即记为命中
- `kqa_pro_qa_contains_instruct`：instruct 模型用的宽松 QA 评测，采用 CoT 风格输出，后续可继续单独迭代 prompt
