# 开发

## 日常

```sh
swift build -c release                 # Engine、Typing、enginecli
Scripts/build.sh                       # build/Typing.app（签名、复制资源）
Scripts/install.sh                     # 装到 ~/Library/Input Methods 并重启输入法进程
Scripts/gate.sh                        # 装机门槛（见下）
Scripts/symtest.sh                     # 按键测试：用假的文本客户端驱动控制器
```

规矩：改一处、测全部、不降才装机。装机版按大众用法，本机学习按个人。

## 评测工具 enginecli

```sh
E=.build/out/Products/Release/enginecli
R=Resources
C=(--model $R/LanguageModel.mlmodelc --vocab $R/vocab.json --lexicon $R/lexicon.sqlite --bigram $R/bigram.bin)

$E $C nihao "women=1"                                 # 看候选；"拼音=2,1" 模拟选词
$E $C --eval Data/eval_word_mixed.tsv                 # 词测试：首选率、首页率、前二率
$E $C --eval Data/eval_word_mixed.tsv --typing --limit 300   # 逐键计时（快速路径 + 后台重排）
$E $C --eval Data/eval_word_mixed.tsv --pace 120 --limit 60  # 按 120 ms 一键经输入会话模拟
$E $C --stress 20                                     # 多线程并发 20 秒，查线程安全
$E $C --graph-scorer $R/weights --learn-demo forum2   # 完整学习会话（GPU），含中途暂停
$E $C --tuned <学习目录>/tuned --compare-backends --eval ... --limit 1   # 写回的 Core ML 模型与 GPU 模型比分数
$E $C --eval FILE --no-rerank --dump-lattice OUT.tsv   # 导出格子读法，给训练侧的评测脚本用
```

评测文件格式：`拼音<TAB>答案<TAB>前文`（前文中的换行写作 `\n`）。`Data/eval_word_{full,mixed,initials,english}.tsv` 是主测试（`Tools/make_word_eval.py`、`Tools/make_english_eval.py` 生成）。

`Tools/syspinyin` 用同样的文件测系统拼音，作为参照。

## 词库与模型的重建

语料、检查点和中间表都在 `Data/`（几十 GB，不入库）。Python 环境在 `Data/venv-train`（PyTorch、coremltools）。完整顺序：

1. `Tools/collect_sogou.py`、`Tools/collect_cells.py`、`Tools/merge_lexicons.py` → `Data/lexicons/all_words.tsv`
2. `Training/prepare_data.py` → `Data/lm*/`（字级 token 流、词表、留出集）
3. `Training/count_words.py` → `Data/lexicon/word_counts.tsv`；`Tools/make_english_casing.py`
4. `Tools/build_lexicon.py` → `Data/lexicon/lexicon.sqlite`；`Tools/build_bigram.py` → `bigram.bin`（词 id 属于这一份词库）
5. `Training/make_words.py`、`Training/make_english.py`、`Training/train.py` → `model.pt`
6. `Scripts/build_model.sh Data/lm9/model.pt` → `Resources/LanguageModel.mlmodelc`、`vocab.json`；`Training/export_weights.py` → `Resources/weights.bin/json`
7. `Tools/make_word_eval.py` 等 → `Data/eval_*.tsv`；`Tools/make_personal_eval.py` → `Data/personal/`

当前装机的模型是 lm9（8 层 512 宽，词表 9026），词库和二元模型为 2026-10-14 的 v3 构建。更换出厂模型后，本机的微调文件会因结构不符自动丢弃并重新学。

## 代码约定

- 引擎与 App 的接口只有 `CandidateDecoding`、`SentenceScorer` 和 `Learner`；App 不直接碰 SQLite 或 Metal。
- 不为具体词写规则；可调的量只在解码器的公开属性里，并能从 enginecli 的参数设置以便测量。
- 新按键行为先在 `Tools/symtest/main.swift` 加用例。
- 工具链注意：macOS 27 SDK 下 `@State` 是宏，在 SwiftPM 构建里从按钮闭包给 `@State` 赋值会报“self 不可变”，界面状态放进 `ObservableObject` 类。
