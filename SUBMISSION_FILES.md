# 最终提交清单

推荐直接以本目录为工程根目录，保持README列出的结构。不要从原code目录再次全量复制历史文件。

## 必须包含

| 文件/目录 | 原因 |
|---|---|
| student.py、model.py | 最终模型和继承的原始GPT；model.py不能删 |
| common.py、evaluate.py | 未修改的官方数据/评分逻辑 |
| data/全部5个文件 | train/validation/test文本、tokenizer.json、manifest.json；官方加载器会核对所有文件 |
| checkpoints/final.pt | 老师无需重新训练即可评分 |
| requirements.txt、README.md | 精确环境和训练/评估教程 |
| REPORT.md | 简短方法、调整过程、对照和局限 |
| train.py、average.py、train_gate.py、configs/final.json | 从头训练最终配方所需 |
| run_pipeline.py | 一键复现完整流程 |
| measure_eval.py、checkpoints/baseline.pt、configs/baseline.json | CPU资源对照与基线复现 |
| tests/、verify_package.py、PACKAGE_MANIFEST.json | 正确性及文件完整性核对 |
| evidence/ | 训练/验证选择、消融、成本和最终分数证据 |
| .gitignore | 防止把本地运行产物混进提交 |

## 不要放入最终工程

- 原`.venv`：体积大且不可跨机器直接使用，应按requirements重新安装。
- `.idea`、`__pycache__`、`.DS_Store`、临时文件。
- 原runs中的全部step权重、未完成训练、旧最终模型及备份。
- SwiGLU/RMSNorm/卷积的实现和训练脚本；最终推理不使用它们。
- 旧绘图工具、重复调参脚本、历史多份README和报告。
- 上传后的访问令牌、账号密码或本机配置。

相关历史对照已压缩为evidence中的JSON/CSV记录；其中出现原实验目录绝对路径只是审计信息，不是运行依赖。
原工程留在上一级code中作为本地档案，未执行不可恢复的历史删除。本提交目录已经排除了无关内容。

## 发布与课程提交

按课程GUIDE，最终提交需要：不可变代码链接，以及匹配权重包的链接，并附不超过10页的报告。

1. 检查REPORT.md中的事实，补充自己的姓名/学号到提交表单或报告首页；当前没有替你虚构身份。
2. 将本目录上传到你自己的仓库；用确切commit链接固定代码版本，避免只给会变化的main分支链接。
3. 权重可直接随工程提供，也可上传为同一版本的Release附件；保留final.pt的SHA256。
4. 最終工程压缩包包含代码、数据和两份权重，可完整离线评分；另有仅权重压缩包便于单独提供第二个链接。
5. 评审解压工程后，执行README第三节，预期BPB为1.5553096072435364。
6. 将REPORT.md导出为PDF时检查实际页数不超过10页；Markdown源稿简短，但导出页数取决于版式，本包不虚报PDF页数。

报告、代码和权重必须对应同一个冻结版本。最终代码经过清理，student.py文件哈希与历史实验源码不同；权重未变化，本次精简后全测试用于确认预测和分数一致。
PACKAGE_MANIFEST.json记录当前提交版本；evidence/original_freeze.json保留历史冻结证据，二者用途不同。
