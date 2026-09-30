# MP1最终提交

本目录是独立提交工程。最终检查点为 `checkpoints/final.pt`，固定评估协议 `7506-mp1-wt2-v2`。
已训练权重的完整 CPU FP32 测试 BPB 为 **1.5553096072435364**，token PPL 为 **25.82342986323253**。BPB 越低越好。

先阅读本教程；简短方法报告见 [REPORT.md](REPORT.md)，提交文件清单见 [SUBMISSION_FILES.md](SUBMISSION_FILES.md)。
`evidence/` 保存历史实验、最终训练参数与本次精简后的评分证据。

## 1. 工程里有什么

```text
FINAL_SUBMISSION/
├── README.md                  本地全过程教程
├── REPORT.md                  简短方法与实验报告
├── SUBMISSION_FILES.md        提交清单、无需提交的文件
├── PACKAGE_MANIFEST.json      文件大小与SHA256
├── requirements.txt          固定依赖版本
├── .gitignore
├── student.py                最终模型（已删除无关实验结构）
├── model.py                  原始GPT；模型基础与资源对照，必须保留
├── common.py                 原始数据、tokenizer、窗口与设备工具
├── evaluate.py               未修改的官方评分器
├── measure_eval.py           Windows峰值内存测量包装器
├── train.py                  主干训练，支持学习率/权重衰减参数
├── average.py                同轨迹权重平均，验证集选择
├── train_gate.py             冻结主干，训练缓存门控
├── run_pipeline.py           一键训练→平均→门控→冻结→CPU测试
├── verify_package.py         文件完整性检查
├── configs/
│   ├── final.json            训练最终主干的配置
│   └── baseline.json         课程基线配置
├── data/                     原始三份文本、tokenizer和manifest
├── checkpoints/
│   ├── final.pt              提交评估用最终权重
│   └── baseline.pt           原始基线权重，用于CPU时间对照
├── tests/                    基线及最终模型正确性检查
└── evidence/                 训练、选择、评分与整理核验证据
```

`checkpoints/final.pt` 内部包含完整推理配置。评估器从权重读取配置，不依赖手动选择 final.json。
final.json 中 `variant=residual_dropout` 是训练主干的接口；最终权重中 `variant=gated_cache` 会启用动态缓存。
权重里的 `cache_lambda=0` 是旧字段；动态缓存实际使用训练得到的门控，绝不表示缓存被关闭。

## 2. 安装 Python 3.12 环境

建议把整个文件夹放在一个固定路径。以下命令在 Windows PowerShell / PyCharm Terminal 中执行。

```powershell
cd E:\MP1_student_starter\MP1_student_starter\FINAL_SUBMISSION
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

如果没有 `py` 命令，先安装 Python 3.12，并用其 python.exe 的完整路径代替 `py -3.12`。
不需要执行 Activate.ps1，直接使用上述解释器路径，避免 PowerShell 激活脚本策略问题。

**只用CPU评分：**

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**有兼容CUDA的NVIDIA显卡，准备训练：**

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

两条安装路径二选一。本次原实验环境为 Python3.12、PyTorch2.7.1+cu126、RTX3060 Laptop；评估均用CPU FP32。
依赖是 torch2.7.1、numpy2.5.3、tokenizers0.21.4；训练不需要 matplotlib、transformers、FAISS 或联网下载数据。
安装依赖需要联网，之后训练与评估使用随包数据，可离线执行。

检查环境：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
.\.venv\Scripts\python.exe verify_package.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

本机整理核验中，12项正确性检查通过。改动包内文件后哈希检查会报告变化，这是预期行为，不要为了掩盖改动随意覆盖原清单。

## 3. 不训练，直接复现提交分数

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints/final.pt --split test --device cpu --precision fp32 --threads 4 --output results/final_test.json
```

预期关键结果：

```json
{"bpb": 1.5553096072435364, "token_ppl": 25.82342986323253, "targets": 428405, "utf8_bytes": 1292013}
```

数据加载可能先等待约20秒；这是正常现象。测试过程不更新模型参数。
CPU浮点实现不同可能产生很小数值差异，耗时和内存也因机器变化；不要用GPU/BF16结果代替提交的CPU FP32分数。

在 Windows 上同时记录完整进程峰值内存：

```powershell
.\.venv\Scripts\python.exe measure_eval.py --checkpoint checkpoints/baseline.pt --split test --device cpu --precision fp32 --threads 4 --output results/baseline_test.json
.\.venv\Scripts\python.exe measure_eval.py --checkpoint checkpoints/final.pt --split test --device cpu --precision fp32 --threads 4 --output results/final_test.json
```

两个命令按顺序执行。`*.json` 是评分，`*.resources.json` 是内存与文件大小，`*.window-nll.npy` 是各窗口损失。
用最终模型JSON中的seconds除以基线seconds，必须≤5；峰值RAM≤4GiB；推理资产≤64MiB。
measure_eval.py 仅支持 Windows 峰值内存测量；其他系统直接使用官方 evaluate.py 评分，并按课程要求另测资源。

## 4. 一键从头训练到测试

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --out runs/reproduce-s17 --device cuda --precision bf16
```

输出目录必须不存在；重新运行请改为 `runs/reproduce-s17-v2`，不要覆盖旧证据。
不支持BF16的CUDA设备用 `--precision fp32`；只有CPU时用 `--device cpu --precision fp32`，耗时会显著增加。
RTX3060 Laptop 上主干训练原记录约9分钟，加上平均、门控和评分通常还需数分钟；这不是所有机器的时间保证。

流程如下：

1. 主干训练12,000步，batch32，每600步在验证集评分并保存检查点。
2. 在最佳验证单点之前，比较最近1/3/5/7/9个同轨迹检查点的平均。
3. 按验证集选中平均主干，冻结其参数，训练缓存门控最多10轮，保存最佳MLP门控。
4. 写入源码与权重哈希的FROZEN.json，随后用CPU FP32全测试。

主要输出：

```text
runs/reproduce-s17/
  train/             主干权重、metrics.json、command.json、源码快照
  average/           平均权重和selection.json
  gate/              门控权重、训练计划和selection.json
  final.pt           本次重新训练选中的最终模型
  FROZEN.json        本次测试前冻结记录
  test_cpu_fp32.json 本次测试分数
  *.log              每个阶段的输出日志
```

一键脚本的进度输出主要显示阶段名，详细步数看 `train.log` 等文件。
训练可重复执行，但不同设备或数值实现可能使从头训练的最优step/epoch及分数略有变化。提交分数对应随包 `checkpoints/final.pt`，不是承诺任何重新训练都会逐位相同。

## 5. 手动逐阶段训练：理解每个步骤

**第一步：主干训练。**

```powershell
.\.venv\Scripts\python.exe train.py --config configs/final.json --run-dir runs/manual-train --device cuda --precision bf16 --threads 4 --seed 17 --steps 12000 --batch-size 32 --eval-every 600 --save-best --lr 0.001 --weight-decay 0.20
```

主干：4层、192维、4头、FFN768维，RoPE theta10000，dropout0，LayerNorm，平方ReLU。
AdamW betas=(0.9,0.999)、100步warmup、余弦最低比例0.1、梯度裁剪1。
处理训练目标：12000×32×256=98,304,000。
`checkpoint.pt` 是训练最后一步，`best_validation.pt` 是最佳单点；它们都还不是带动态缓存的最终提交模型。

**第二步：权重平均。**

```powershell
.\.venv\Scripts\python.exe average.py --run-dir runs/manual-train --out runs/manual-average --device cuda
```

查看 `runs/manual-average/selection.json` 中 `selected.checkpoint`。
原实验选中9点：3600、4200、4800、5400、6000、6600、7200、7800、8400。
只平均同一条训练轨迹；不要平均不同种子或不同模型结构。
保留训练总步数12000：虽然最终选中8400之前的权重，余弦学习率仍按12000步计算，直接把总步数改成8400会改变配方。

**第三步：训练缓存门控。**

```powershell
.\.venv\Scripts\python.exe train_gate.py --parent runs/manual-average/average-9.pt --out runs/manual-gate --device cuda --epochs 10 --lr 0.001
```

上面average-9.pt须替换为第二步实际选中的检查点。parent必须是未带门控的平均主干，不能直接传入checkpoints/final.pt。
该阶段用训练文本拟合161个MLP门控参数，冻结主干；门控weight_decay0.01、batch4096。
脚本保留一个learned constant对照，以复现原实验比较；最终checkpoint.pt取验证最好的MLP门控。
原实验最佳为第9轮，最大搜索10轮。不要盲目取第10轮。

**第四步：验证，然后冻结并测试。**

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/manual-gate/checkpoint.pt --split validation --device cpu --precision fp32 --threads 4 --output runs/manual-gate/validation_cpu.json
```

原提交权重验证BPB为1.539103594109。选择只能依据验证集；确认代码、配置和权重后记录SHA256，再测试：

```powershell
Get-FileHash student.py,model.py,common.py,evaluate.py,runs/manual-gate/checkpoint.pt -Algorithm SHA256
.\.venv\Scripts\python.exe measure_eval.py --checkpoint runs/manual-gate/checkpoint.pt --split test --device cpu --precision fp32 --threads 4 --output runs/manual-gate/test_cpu.json
```

不要根据测试分数反复调整参数。若只想交作业复现已报告分数，直接使用第三节，无需重训。

## 6. PyCharm 绿色按钮配置

打开 FINAL_SUBMISSION 文件夹作为项目，设置解释器为本目录 `.venv\Scripts\python.exe`。
“运行→编辑配置→添加Python”，工作目录始终设为 FINAL_SUBMISSION。

| 用途 | 脚本 | Parameters |
|---|---|---|
| 直接测提交模型 | evaluate.py | `--checkpoint checkpoints/final.pt --split test --device cpu --precision fp32 --threads 4 --output results/pycharm_test.json` |
| 一键重新训练 | run_pipeline.py | `--out runs/pycharm-reproduce --device cuda --precision bf16` |
| 单独训练主干 | train.py | 使用第五节第一条命令中脚本名之后的全部参数 |

不要把 student.py 当成训练入口，它只定义网络。主干训练入口是 train.py，完整流程入口是 run_pipeline.py。

## 7. 常见问题

- `Run directory already contains results`：换一个新的runs输出目录，不要覆盖先前实验。
- `CUDA is unavailable`：确认使用CUDA版torch和NVIDIA驱动，或者切换CPU；依赖安装与实验失败不等于模型分数错误。
- `ModuleNotFoundError: common`：工作目录或工程文件缺失；从FINAL_SUBMISSION运行命令，不要单独复制student.py。
- `Changed benchmark file`：数据哈希不符。重新解压原包，不要编辑文本或重新训练tokenizer。
- checkpoint缺少/多出键：选择了历史其他结构的权重。这个精简包只支持最终RoPE结构与原始基线。
- `verify_package`失败：检查是否改了原文件。新增results和runs不影响已登记文件的哈希。

## 8. 来源与数据许可

方法来源与讨论见REPORT.md。保留课堂model.py/common.py/evaluate.py，不擅自给课程代码添加新的开源许可。
WikiText-2来源于Salesforce WikiText、文本来自Wikipedia贡献者。上游标识CC BY-SA 3.0和GFDL；再分发应保留署名及许可链接：
[数据集](https://huggingface.co/datasets/Salesforce/wikitext)、[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)、[GFDL](https://www.gnu.org/licenses/fdl-1.3.html)。
原数据revision为b08601e04326c79dfdd32d625aee71d232d685c3；按行连接UTF-8，tokenizer仅用训练文本拟合。数据哈希见data/manifest.json。

本目录已经整理，但不表示已替你上传、公开或提交课程网站。
