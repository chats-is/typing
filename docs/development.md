# 开发

四件事都能在这个目录里完成：改代码、更新词库、更新数据、训练模型，每一步都有测试。

## 日常

```sh
swift build -c release                 # Engine、Typing、enginecli
Scripts/build.sh                       # build/Typing.app（签名、复制资源）
Scripts/install.sh                     # 装到 ~/Library/Input Methods 并重启输入法进程
Scripts/gate.sh [私有回放文件]          # 装机门槛：四项词测试、每键计时、节奏模拟、并发压力、按键测试
Scripts/symtest.sh                     # 只跑按键测试
```

规矩：改一处、测全部、不降才装机。装机版按大众用法，本机学习按个人。

每一次改动都走同一条路：

1. 只改一处（代码、词库、数据或模型里的一项）。
2. `Scripts/gate.sh`：四项词测试、五个句子集、一次学习会话、每键计时、节奏模拟、并发压力、按键测试。有私有回放文件就作为参数附上。
3. 对照 `docs/benchmarks.md`，任何一项低于基准就不装、不提交，回去改。
4. 通过后 `Scripts/install.sh` 装到本机，用一段时间。
5. 提交，并把变化写进 `docs/benchmarks.md`（数字）或 `docs/decisions.md`（新决定）。
6. 推到 GitHub 后 CI 再跑一遍；打 `v*` 标签即发布。

按改动类型再加专项：按键和界面改动先在 `Tools/symtest/main.swift` 加用例；词库或二元模型改动看句子集和 `--show-failures` 的失败样例；学习改动看 `--learn-demo` 的前后数字和回滚是否正确；性能改动看 `--typing` 和 `--pace`。

## 目录里的数据（`Data/`，不入库）

| 位置 | 内容 | 由谁生成 |
|---|---|---|
| `corpus2/`、`corpus3/` | 全部训练与统计语料（来源见 `Training/sources.py`、`Data/lexicons/sources.tsv`） | 下载 |
| `lexicons/` | 各家公开词库原文与合并表 `all_words.tsv` | `Tools/collect_*.py`、`merge_lexicons.py` |
| `lexicon/` | 正式词库产物：`lexicon.sqlite`、`bigram.bin`、`word_counts_vit3.tsv`、`english_casing.tsv`、`readings.tsv`、`lm0.tsv` | 下面的词库流程 |
| `lm9/` | 当前模型：`model.pt`、各来源 token 流、`vocab.json`、`sources.json`、32 行导出、`weights.*` | 下面的模型流程 |
| `base/lm3_model.pt` | lm9 的起点（从头训练的 8 万步基座） | 历史产物 |
| `holdout/` | 各来源留出文本，所有评测集的来源 | `Training/prepare_data.py` |
| `eval_*.tsv`、`wordlattice/`、`personal/` | 评测集、词测试的格子导出、学习验证材料 | 下面的评测流程 |
| `venv-train/` | Python 环境（PyTorch、coremltools、pypinyin） | `python -m venv` + pip |

`Resources/` 里的三个大文件就是 `lexicon/lexicon.sqlite`、`lexicon/bigram.bin`、`lm9/LanguageModel32.mlmodelc` 的拷贝；`positions.bin` 由 `Tools/extract_positions.py` 从 `lm9/weights` 取出（完整 64 行位置表），`weights.json` 只保留结构参数。

## 评测

```sh
E=.build/out/Products/Release/enginecli
R=Resources
C=(--model $R/LanguageModel.mlmodelc --vocab $R/vocab.json --lexicon $R/lexicon.sqlite --bigram $R/bigram.bin)

$E $C nihao "women=1"                                 # 看候选；"拼音=2,1" 模拟选词
$E $C --eval Data/eval_word_mixed.tsv                 # 词测试：首选率、首页率、前二率
$E $C --eval Data/eval_dialogue.tsv --limit 300       # 句子集
$E $C --eval Data/eval_word_mixed.tsv --typing --limit 300   # 逐键计时
$E $C --eval Data/eval_word_mixed.tsv --pace 120 --limit 60  # 按 120 ms 一键经输入会话模拟
$E $C --stress 20                                     # 多线程并发，查线程安全
$E $C --graph-scorer $R/weights --learn-demo forum2   # 完整学习会话（GPU），含中途暂停
$E $C --tuned <学习目录>/tuned --compare-backends --eval Data/eval_word_full.tsv --limit 1   # 写回的 Core ML 与 GPU 模型比分数
$E $C --eval FILE --no-rerank --dump-lattice OUT.tsv   # 导出格子读法，给训练侧的评测脚本用
```

评测文件格式：`拼音<TAB>答案<TAB>前文`（前文中的换行写作 `\n`）。`Tools/syspinyin` 用同样的文件测系统拼音，结果在 `Data/eval_sys_*.tsv`。

重新生成评测集（都从 `Data/holdout/` 出发，模型从未见过这些文本）：

```sh
P=Data/venv-train/bin/python
$P Tools/make_word_eval.py              # 主测试：eval_word_{full,mixed,initials}.tsv
$P Tools/make_english_eval.py           # eval_word_english.tsv / eval_english_*.tsv
$P Tools/make_genre_eval.py             # 各体裁句子集 eval_<source>.tsv
$P Tools/make_input_eval.py             # eval_abbr_mix.tsv、eval_typo_mix.tsv
$P Tools/make_context_eval.py           # eval_context.tsv
$P Tools/make_personal_eval.py --source forum --rank 2 --out Data/personal/forum2   # 学习验证材料
```

## 更新词库

```sh
P=Data/venv-train/bin/python
$P Tools/collect_sogou.py; $P Tools/collect_cells.py        # 采集（可选，已有 Data/lexicons）
$P Tools/merge_lexicons.py                                   # → Data/lexicons/all_words.tsv
$P Tools/count_words_viterbi.py                              # → Data/lexicon/word_counts_vit3.tsv（概率分词计数，3 轮）
$P Tools/make_english_casing.py                              # → Data/lexicon/english_casing.tsv
$P Tools/build_lexicon.py                                    # → Data/lexicon/lexicon.sqlite
$P Tools/build_bigram.py                                     # → Data/lexicon/bigram.bin（词 id 绑定这份词库；词对计数缓存在 bigram_counts.npz，--from-counts --min-count N 可快速重写）
cp Data/lexicon/lexicon.sqlite Data/lexicon/bigram.bin Resources/
Scripts/gate.sh
```

词库规则在 `Tools/build_lexicon.py` 顶部：多词库共同收录的保底频率、单字按读音拆行、英文词的常见大小写。词频只来自真实文本的概率分词，不手工设定。

## 更新数据、训练模型

```sh
P=Data/venv-train/bin/python
$P Training/prepare_data.py --out Data/prepared            # 语料 → token 流、词表、留出集
$P Training/make_words.py --out Data/prepared              # 词条行（让模型会读孤立词）
$P Training/make_english.py --out Data/prepared            # 英文词行（含大小写）
$P Training/train.py --data Data/prepared --out Data/prepared/model.pt \
    --init Data/base/lm3_model.pt --lr 5e-5 --steps 6000   # lm9 的训法：从 lm3 基座继续训练
$P Training/eval_checkpoint.py --model Data/prepared/model.pt --vocab Data/prepared/vocab.json \
    Data/wordlattice/word_mixed.lattice.tsv Data/wordlattice/word_full.lattice.tsv   # 不导出先看主指标
LM_ROWS=32 Scripts/build_model.sh Data/prepared/model.pt   # → Resources/LanguageModel.mlmodelc、vocab.json
$P Training/export_weights.py Data/prepared/model.pt Data/prepared/weights   # → weights.bin/json（开发用）
python3 Tools/extract_positions.py Data/prepared/weights Resources/positions.bin   # 位置表；weights.json 只需 config
Scripts/gate.sh
```

模型：8 层、512 宽、8 头、窗口 64 的字级 GPT，词表 9026，2990 万参数（`Training/model.py`）。从头训练用 `train.py --steps 80000`（无 `--init`）。换了出厂模型后，本机的微调文件会因结构不符自动丢弃并重新学习；`Data/lexicon/lm0.tsv` 来自 `Tools/add_lm0.py` 时代，建库脚本仍写这一列但引擎不读它。

当前装机：模型 lm9（lm3 基座 + 6000 步继续训练，验证损失 3.130，32 行导出），词库与二元模型为 2026-10-14 11:21 的构建。

## 代码约定

- 引擎与 App 的接口只有 `CandidateDecoding`、`SentenceScorer` 和 `Learner`；App 不直接碰 SQLite 或 Metal。
- 不为具体词写规则；可调的量只在解码器的公开属性里，并能从 enginecli 的参数设置以便测量。
- 新按键行为先在 `Tools/symtest/main.swift` 加用例。
- 工具链注意：macOS 27 SDK 下 `@State` 是宏，在 SwiftPM 构建里从按钮闭包给 `@State` 赋值会报“self 不可变”，界面状态放进 `ObservableObject` 类。

## 归档

`../typing-archive/` 放着定稿前的全部实验：旧模型 lm 到 lm8、lm9x、知乎语料、旧引擎、按键与读音模型实验、词库的分词方案对比、旧评测集和日志。后续开发不需要它们，可以删除。
