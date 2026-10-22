# 开发

四件事都能在这个目录里完成：改代码、更新词库、更新数据、训练模型，每一步都有测试。

## 日常

```sh
swift build -c release                 # Engine、Typing、enginecli
Scripts/build.sh                       # build/Typing.app（签名、复制资源）
Scripts/install.sh                     # 装到 /Library/Input Methods（要一次管理员密码）并重启输入法进程
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

用户报告的输入问题记在 `Data/eval_reported.tsv`（每个一行：拼音、期望的词、前文），修的时候用它看结果：

```sh
.build/release/enginecli --model Resources/LanguageModel.mlmodelc --vocab Resources/vocab.json \
  --lexicon Resources/lexicon.sqlite --bigram Resources/bigram.bin --eval Data/eval_reported.tsv --show-failures
```

| 记录日期 | 拼音 | 期望 | 现在 | 原因 |
|---|---|---|---|---|
| 2026-10-18 | dialog | 英文候选 dialog | 已解决：第一个候选 dialog | 两个原因都在词库数据：英文词频表只取前 3 万词（dialog 排第 30421），改为取满 5 万；每个词只给中文文本里最常见的写法（Dialog），改为普通词默认小写、其他常见写法作为后面的候选 |

按改动类型再加专项：按键和界面改动先在 `Tools/symtest/main.swift` 加用例；词库或二元模型改动看句子集和 `--show-failures` 的失败样例；学习改动看 `--learn-demo` 的前后数字和回滚是否正确；性能改动看 `--typing` 和 `--pace`。

## 目录里的数据（`Data/`，不入库）

| 位置 | 内容 | 由谁生成 |
|---|---|---|
| `corpus2/`、`corpus3/` | 全部训练与统计语料（来源见 `Training/sources.py`、`Data/lexicons/sources.tsv`） | 下载 |
| `lexicons/` | 各家公开词库原文与合并表 `all_words.tsv` | `Tools/collect_*.py`、`merge_lexicons.py` |
| `lexicon/` | 正式词库产物：`lexicon.sqlite`、`bigram.bin`、`bigram_counts.npz`（词对计数缓存）、`word_counts_vit3.tsv`、`english_casing.tsv`、`readings.tsv` | 下面的词库流程 |
| `prepared-clean/` | 清洗后的数据：各来源 token 流、`vocab.json`、`sources.json`（第三段的配比）、清洗后的留出文本 `holdout_*.txt` 与验证集 `val.bin`；`holdout-raw/`、`val.raw.bin` 是清洗留出文本之前的原样备份 | `Training/prepare_data.py` |
| `clean2/` | 当前模型：`model.pt`（三段训练的结果）、`a.log`/`b.log`/`c.log`、`weights.*`；导出在 `build/clean2/` | `Training/train_base.sh` |
| `fresh/` | 判定模型用的新测试集和学习检查集的句子（入库） | `Tools/make_fresh_eval.py` |
| `lm9/` | 上一个模型（未清洗的数据）：`model.pt`、各来源 token 流（9 GB）、`vocab.json`、`sources.json`、32 行导出、`weights.*` | 历史产物 |
| `base/lm3_model.pt` | lm9 的起点：lm2（从头 8 万步）接着训 2 万步（学习率 2e-4，加入网页、技术文档、tatoeba 和英文大小写）；lm9 = 它再训 6000 步 | 历史产物 |
| `holdout/` | 各来源原样的留出文本，旧评测集的来源 | `Training/prepare_data.py`（旧版） |
| `eval_*.tsv`、`wordlattice/`、`personal/` | 评测集、词测试的格子导出、学习验证材料 | 下面的评测流程 |
| `venv-train/` | Python 环境（PyTorch、coremltools、pypinyin） | `python -m venv` + pip |

`Resources/` 里的三个大文件就是 `lexicon/lexicon.sqlite`、`lexicon/bigram.bin`、`build/clean2/LanguageModel.mlmodelc` 的拷贝；`positions.bin` 由 `Tools/extract_positions.py` 从 `clean2/weights` 取出（完整 64 行位置表），`weights.json` 只保留结构参数（`{"config": {...}}`，不含 `tensors`）。

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

词频、二元模型和英文大小写，都从 `Data/prepared-clean` 的清洗后 token 流统计（2026-10-22 起的默认目录）。现在装的词库和二元模型是 2026-10-14 从未清洗的 `Data/lm9` 统计的，下次重建时就换成清洗后的统计。重建以后 `bigram_counts.npz` 缓存要一起重算，不能用 `--from-counts`。

## 更新数据、训练模型

清洗、训练、测试的完整流程，2026-10-22 定的出厂模型 clean2 就是这样做出来的。

### 1. 清洗数据

规则在 `Training/cleaning.py`，训练文本和留出文本用同一套规则；`Training/test_cleaning.py` 逐条检查（该去掉的，以及不能误伤的正常打字）。
- **个人信息换成“＊”**（`redact()`，与输入法学习记录的 `Engine/Redaction.swift` 完全一致）：
  - 邮箱；
  - 网址，包括：紧跟在汉字、数字或字母后的网址，语料里丢了“://”的网址（httpwww.bilibili.com…），以及 yys.163.com 这样的小写域名；
  - 6 位以上连续数字（电话、身份证、银行卡），分组写的卡号和手机号；
  - 账号：微信、QQ、抖音、快手、支付宝、淘宝、旺旺、钉钉、小红书、Line、Skype、Telegram 后面的账号。英文平台名后面要有“号”或冒号，只隔空格时账号里要有数字，所以 wxWidgets、“go on QQ immediately”不算；
  - “姓名、联系人、收件人”后面的名字，“@”后面的人（变成 @＊）；
  - 住址里的楼栋、单元和房号（11楼1211、3号楼2单元401、4603室），车牌；
  - 字母数字混排、20 位以上、数字多的乱码串：链接参数、校验码、图片文件名。数字只占一两位的名字保留，如 CanvasRenderingContext2D、RenaultMeganeRS275。
  - 日期、时间、价格、街道名、普通数字都保留。
- **去掉页面结构**：HTML 转义字符换回原字符；去掉以下内容，不论它们是单独一行，还是写在正文中间：
  - HTML 标签，包括语料里被截断的（“<img data-rawwidth正文…”“<a class”、行末的“</p”）；
  - 论坛排版代码（[size=14]）；
  - 论坛回帖抬头：“作者：… 回复日期：…”“@某人 3楼 2015-09-09 08:18:00”“某人 3楼 …”“发布者：… 发表日期：2013年05月21日”“来自：…客户端”。
- **去掉不是打出来的行**：8 字以上、数字和符号超过 60%、汉字和字母不到 30% 的行，例如表格、分隔线、时间戳。
- **整篇丢掉**：200 字以上、数字超过 20% 的文档，例如价目表、统计表。

改规则时：先在 `test_cleaning.py` 里加用例；再拿真实语料对比新旧规则，逐条看被改动的地方；Swift 那边同步修改，并用同一批句子比对两边的结果。

```sh
P=Data/venv-train/bin/python
$P Training/prepare_data.py --out Data/prepared-clean --cased-from Data/lm9 > Data/prepared-clean/prepare.log   # 全部来源 → 清洗 → token 流、留出文本、验证集（约 25 分钟）
$P Training/make_words.py --out Data/prepared-clean                                # 词条行
$P Training/make_english.py --out Data/prepared-clean --data Data/prepared-clean   # 英文词行（含大小写）
cp Data/lm9/sources.json Data/prepared-clean/sources.json                         # 第三段的配比
```

`--cased-from Data/lm9` 沿用 lm9 的词表（9026 个字符，含 26 个大写字母），换模型后词表不变。每次准备的结果都在 `Data/prepared-clean/prepare.log`，里面有各来源清洗删去的字数。

数据版本：
- **clean2 的数据**（2026-10-18，旧规则）：训练数据的副本在 mini 的 `Data/clean1`，校验值、日志、配比、原样留出文本在 `Data/clean2/data/`，当时的规则是 `cleaning-commit.txt` 里那个提交的 `Training/cleaning.py`。清洗删去的字数：对话 0.05%、混合 1.08%、新闻 1.86%、论坛 9.23%、网页 0.93%，其余来源不到 0.5%。
- **2026-10-22 起的新规则**：修了旧规则的两个问题。
  - 紧跟在汉字后的网址没被替换。另一方面，网址规则遇到汉字不停，会把网址后面整段中文一起删掉，新规则在汉字前停下。
  - 正文中间的回帖抬头、带字母的用户名、微信号、门牌房号、车牌、页面标签都留在了数据里。

2026-10-22 以前的版本不清洗留出文本。已经生成的目录用 `--clean-holdouts` 就地补洗一次，不重新处理语料：

```sh
$P Training/prepare_data.py --out Data/prepared-clean --clean-holdouts
```

### 2. 训练（Mac mini）

```sh
rsync -a Data/prepared-clean/ john@John-Mac-mini.local:typing/Data/clean1/
ssh john@John-Mac-mini.local 'cd ~/typing && caffeinate -i Training/train_base.sh Data/clean1 Data/run1'
```

`Training/train_base.sh` 是完整的三段做法，约 36 小时：
1. 从头训 8 万步，用 lm2 的配比；
2. 接着训 2 万步，学习率 2e-4，用 lm3 的配比；
3. 再训 6000 步，学习率 5e-5，用数据目录里 `sources.json` 的配比。

只用第三段的配比从头训，各项都更差（2026-10-20 的 clean1）。clean2 三段结束时的验证损失：3.202 → 3.138 → 3.143。这些是在清洗前的原样验证集上算的，和 lm9 的 3.130 不能直接比：在原样文本上 clean2 差，差别全在网页和论坛的表格、网址里；同样的文本清洗之后，clean2 反而更好。

### 3. 导出

```sh
rsync -a john@John-Mac-mini.local:typing/Data/run1/model.pt Data/run1/
cp Data/prepared-clean/vocab.json Data/run1/vocab.json
LM_ROWS=32 Scripts/build_model.sh Data/run1/model.pt build/run1          # Core ML（32 行）
$P Training/export_weights.py Data/run1/model.pt Data/run1/weights        # GPU 权重
python3 Tools/extract_positions.py Data/run1/weights build/run1/positions.bin
```

装机时把 `build/run1/LanguageModel.mlmodelc` 和 `positions.bin` 拷进 `Resources/`。`weights.json` 只有结构参数，结构不变就不用动。

### 4. 测试与判定

先在临时目录里拼一份 `Resources/`：除模型和位置表换成新的，其余链接到仓库里的文件。在那个目录里跑 `Scripts/gate.sh` 的各项，再和现在的模型逐句比较：

```sh
$E $C --eval Data/fresh/eval_dialogue.tsv --show-failures > new_dialogue.out   # 两个模型各跑一次
Tools/compare_failures.py old_dialogue.out new_dialogue.out                    # 新错、新对的句数和 p
```

- **判定只看新测试集** `Data/fresh/eval_*.tsv`：句子来自清洗后的留出文本，训练数据里一条都没有，也从来没有用来调过参数。p 低于 0.05 才算真有差别，两次训练之间差几句是正常的。
- **旧句子集**（`Data/eval_dialogue.tsv` 等）只用来防退步：引擎参数是在它们上面调的，lm9 也是在它们上面挑出来的，有不少句子原样出现在训练数据里（对话 84/300，触宝 294/300），换模型时会偏向旧模型。2026-10-22 起，它们的上文也按同样规则清洗过，目标句不受影响。
- **学习检查**：`Resources/general_lattices.tsv` 是 `Data/fresh/general_dialogue.tsv` 的 456 句对话，导出了格子读法，和判定集来自不同的文档。输入法和 `--learn-demo` 用的是这同一份。

重新生成新测试集和学习检查集：

```sh
$P Tools/make_fresh_eval.py
$E $C --eval Data/fresh/general_dialogue.tsv --no-rerank --dump-lattice Resources/general_lattices.tsv
```

2026-10-22 lm9 与 clean2 的比较：
- 新测试集合计 clean2 新错 66 句、新对 62 句，p = 0.79，打字一样。分项里只有论坛 18 比 5，p = 0.01，待观察：上一版新测试集上论坛是 9 比 9。
- 学习测试两个都保留：clean2 用学习率 6e-5 一次通过；lm9 在 6e-5 时普通输入下降，退到 2e-5 才保留。
- 旧学习检查集就是旧对话测试集，在它上面 clean2 被丢弃了。

模型：8 层、512 宽、8 头、窗口 64 的字级 GPT，词表 9026，2990 万参数（`Training/model.py`）。换了出厂模型后，本机的微调文件因模型指纹不符自动作废，再从打字记录重新学。

当前装机：模型 clean2（清洗后的数据，三段训练，32 行导出），词库与二元模型是 2026-10-14 11:21 的构建。上一个模型 lm9 的做法：lm2（从头 8 万步）→ lm3（+2 万步）→ lm9（+6000 步），训练数据没有清洗。

2026-10-16 在本机把流程各段重新走了一遍：
- 建库脚本再生的词库与装机版逐行一致；二元模型重算得到同样的 15,746,631 个词对。
- `build_model.sh` 从 `lm9/model.pt` 再导出的 Core ML 权重与装机版逐字节一致，`export_weights.py`、`extract_positions.py` 的输出也一样。
- 评测集生成脚本再生的文件与入库文件相同。

## 代码约定

- 引擎与 App 的接口只有 `CandidateDecoding`、`SentenceScorer` 和 `Learner`；App 不直接碰 SQLite 或 Metal。
- 不为具体词写规则；可调的量只在解码器的公开属性里，并能从 enginecli 的参数设置以便测量。
- 新按键行为先在 `Tools/symtest/main.swift` 加用例。
- 工具链注意：macOS 27 SDK 下 `@State` 是宏，在 SwiftPM 构建里从按钮闭包给 `@State` 赋值会报“self 不可变”，界面状态放进 `ObservableObject` 类。

## 归档

`../typing-archive/` 放着定稿前的全部实验：旧模型 lm 到 lm8、lm9x、知乎语料、旧引擎、按键与读音模型实验、词库的分词方案对比、旧评测集和日志。后续开发不需要它们，可以删除。
