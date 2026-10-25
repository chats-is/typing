# 开发

四件事都能在这个目录里完成：改代码、更新词库、更新数据、训练模型，每一步都有测试。

## 从拿到代码开始

需要：M 系列芯片的 Mac，macOS 15 以上；Xcode 命令行工具；Python 3.12；git-lfs（`Resources/` 里的三个大文件用 Git LFS 存放）。

```sh
git lfs install && git clone <仓库地址> typing && cd typing
Scripts/build.sh && Scripts/gate.sh                     # 代码和 Resources 完整：能构建，门槛通过
python3.12 -m venv Data/venv-train && Data/venv-train/bin/pip install -r Training/requirements.txt
Data/venv-train/bin/python Tools/fetch_data.py          # 按 Data/manifest.tsv 下载全部原始数据（约 20 GB），逐个核对
```

之后按要做的事走下面的流程：更新词库、更新数据和训练模型、重新生成评测集。每一步的输入，不是仓库里的文件，就是 `Data/manifest.tsv` 登记的原始数据，或者前面某一步的产物。

`Data/manifest.tsv` 一行一份原始数据：在 `Data/` 下的路径、下载方式、地址、sha256、获取日期。地址都是固定版本（GitHub 的提交、Hugging Face 的版本、发布附件、git 仓库的提交）；下载方式是 `latest` 的几类（搜狗、百度细胞词库，Tatoeba）只提供当前版本，下载到的可能比登记的新，`Tools/fetch_data.py` 会报告“已更新”，不算失败。

```sh
python3 Tools/fetch_data.py --check            # 只核对本机数据，不下载
Data/venv-train/bin/python Tools/fetch_data.py --only corpus2   # 只下载路径以此开头的
Data/venv-train/bin/python Tools/fetch_data.py --update         # 有意换成来源现在的版本：重新下载不一致的，并把新的 sha256 写进清单
```

加新的原始数据：放进 `Data/` 后在清单里加一行（下载方式、固定版本的地址、sha256）；`Tools/collect_sogou.py`、`collect_cells.py` 采集的细胞词库会自动登记。

## 日常

```sh
swift build -c release                 # Engine、Typing、enginecli
Scripts/build.sh                       # build/app.noindex/Typing.app（签名、复制资源；放在系统不登记应用的目录里，路径只写在 Scripts/app_path.sh）
Scripts/install.sh                     # 装到 /Library/Input Methods（要一次管理员密码）并重启输入法进程
Scripts/gate.sh [私有回放文件]          # 装机门槛：全部测试，最后自动判定 GATE PASSED / GATE FAILED
Scripts/symtest.sh                     # 只跑按键测试
```

规矩：改一处、测全部、不降才装机。装机版按通用文本，学习按学习记录。

每一次改动都走同一条路：

1. 只改一处（代码、词库、数据或模型里的一项）。
2. `Scripts/gate.sh`：四项词测试、五个防退步句子集、七个判定集、一次学习会话、每键计时、节奏模拟（跑三次）、并发压力、按键测试。有私有回放文件就作为参数附上（只显示，不判定）。最后由 `Scripts/gate_check.py` 自动判定，不用人工对数字：
   - 准确率：每一套都不能低于 `Scripts/gate_baseline.tsv`（和 `docs/benchmarks.md` 是同一组数字），允许差 1 句，因为同一命令重跑有时会差 1 条；
   - 学习会话必须保留，压力测试不能出错，按键测试必须全过；
   - 速度用固定上限判，不和上一次比：1–6 个字母每键平均不超过 16 ms（一帧）；任何长度最慢的 5% 不超过 50 ms；模型重排最慢的 5% 不超过 100 ms。节奏模拟取三次的中位数。同一份代码会因为机器忙闲测出 2.4–7.3 ms，所以开头会打印当时的负载。在 CI（共用的云机器）上速度只显示、不判；
   - 有任何一项不过就打印 `GATE FAILED`，退出码为 1。

   换上新模型、确认要用它以后，把它的数字同时写进 `Scripts/gate_baseline.tsv` 和 `docs/benchmarks.md`。
3. 对照 `docs/benchmarks.md`，任何一项低于基准就不装、不提交，回去改。
4. 通过后用 `Scripts/install.sh` 装机，在真实应用里打字验证。
5. 提交，并把变化写进 `docs/benchmarks.md`（数字）或 `docs/decisions.md`（规则）。
6. 推到 GitHub 后 CI 再跑一遍；打 `v*` 标签即发布。

报告的输入问题记在 `Data/eval_reported.tsv`（每个一行：拼音、期望的词、前文），修的时候用它看结果：

```sh
.build/release/enginecli --model Resources/LanguageModel.mlmodelc --vocab Resources/vocab.json \
  --lexicon Resources/lexicon.sqlite --bigram Resources/bigram.bin --eval Data/eval_reported.tsv --show-failures
```

按改动类型再加专项：按键和界面改动先在 `Tools/symtest/main.swift` 加用例；词库或二元模型改动看句子集和 `--show-failures` 的失败样例；学习改动看 `--learn-demo` 的前后数字和回滚是否正确；性能改动看 `--typing` 和 `--pace`。

## 目录里的数据（`Data/`）

入库的只有：来源清单 `manifest.tsv`、读音表 `lexicon/readings.tsv`、全部评测集 `eval*.tsv`、`fresh/` 的判定集和学习检查集、`personal/` 的学习验证材料。其余都由下面的流程生成或下载。

| 位置 | 内容 | 由谁生成 |
|---|---|---|
| `manifest.tsv` | 全部原始数据的来源、版本和 sha256（入库） | 新增数据时登记 |
| `corpus2/`、`corpus3/` | 全部训练与统计语料（读法见 `Training/sources.py`） | `Tools/fetch_data.py` |
| `lexicons/` | 各家公开词库原文与合并表 `all_words.tsv` | 原文：`Tools/fetch_data.py`（新的细胞词库：`Tools/collect_*.py`）；合并表：`merge_lexicons.py` |
| `lexicon/` | 正式词库产物：`lexicon.sqlite`、`bigram.bin`、`bigram_counts.npz`（词对计数缓存）、`word_counts_vit3.tsv`、`english_casing.tsv`；字的读音表 `readings.tsv`（入库，建词库、准备数据、生成评测集都读它） | 下面的词库流程 |
| `prepared-clean/` | 当前训练数据（clean3 的数据）：清洗、去重、按分数过滤后的各来源 token 流，`vocab.json`、`sources.json`，清洗后的留出文本 `holdout_*.txt` 与验证集 `val.bin`；`prepare.log`、`dedup.log`、`quality.log` 记着每一步删了多少，`*.quality.npy` 是每篇的分数 | 下面的第 1–3 步 |
| `clean3/` | 当前模型：`model.pt`（三段训练的结果）、`a.log`/`b.log`/`c.log`、`weights.*`；导出在 `build/clean3/` | `Training/train_base.sh` |
| `fresh/` | 判定模型用的判定集和学习检查集的句子（入库） | `Tools/make_fresh_eval.py` |
| `eval*.tsv`、`personal/` | 评测集（`eval_sys_*` 是系统拼音的结果）、学习验证材料（入库）；`Tools/make_fresh_eval.py` 生成判定集时避开这些评测集里的句子 | 下面的评测流程 |
| `wordlattice/` | 词测试的格子导出，`Training/eval_checkpoint.py` 读 | 下面的评测流程 |
| `venv-train/` | Python 环境（PyTorch、coremltools、pypinyin） | `python -m venv` + pip |

`Resources/` 里的三个大文件就是 `lexicon/lexicon.sqlite`、`lexicon/bigram.bin`、`build/clean3/LanguageModel.mlmodelc` 的拷贝；`positions.bin` 由 `Tools/extract_positions.py` 从 `clean3/weights` 取出（完整 64 行位置表），`weights.json` 只保留结构参数（`{"config": {...}}`，不含 `tensors`）。

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

重新生成评测集（都从 `Data/prepared-clean/holdout_*.txt` 出发：清洗后的留出文本，模型从未见过）：

```sh
P=Data/venv-train/bin/python
$P Tools/make_word_eval.py              # 主测试：eval_word_{full,mixed,initials}.tsv
$P Tools/make_english_eval.py           # eval_word_english.tsv / eval_english_*.tsv
$P Tools/make_genre_eval.py             # 各体裁句子集 eval_<source>.tsv
$P Tools/make_input_eval.py             # eval_abbr_mix.tsv、eval_typo_mix.tsv
$P Tools/make_context_eval.py           # eval_context.tsv
$P Tools/make_personal_eval.py --source forum --rank 2 --out Data/personal/forum2   # 学习验证材料
for t in full mixed initials; do $E $C --eval Data/eval_word_$t.tsv --no-rerank --dump-lattice Data/wordlattice/word_$t.lattice.tsv; done   # 检查点评测用的格子
```

## 更新词库

```sh
P=Data/venv-train/bin/python
$P Tools/collect_sogou.py; $P Tools/collect_cells.py        # 采集（可选，已有 Data/lexicons）
$P Tools/merge_lexicons.py                                   # → Data/lexicons/all_words.tsv
$P Tools/count_words_viterbi.py                              # → Data/lexicon/word_counts_vit3.tsv（概率分词计数，3 轮；第一轮的初始频率取自 --start，默认 Resources/lexicon.sqlite）
$P Tools/make_english_casing.py                              # → Data/lexicon/english_casing.tsv
$P Tools/build_lexicon.py                                    # → Data/lexicon/lexicon.sqlite
$P Tools/build_bigram.py                                     # → Data/lexicon/bigram.bin（词 id 绑定这份词库；词对计数缓存在 bigram_counts.npz，--from-counts --min-count N 可快速重写）
cp Data/lexicon/lexicon.sqlite Data/lexicon/bigram.bin Resources/
Scripts/gate.sh
```

词频是迭代计算的，结果和第一轮的初始频率有关：默认从装机的词库出发，每次更新都接着上一版；要重现某一版的词频，用 `--start` 指定它出发时的那份词库。

词库规则在 `Tools/build_lexicon.py` 顶部：多词库共同收录的保底频率、单字按读音拆行、英文词的常见大小写。词频只来自真实文本的概率分词，不手工设定。

词频、二元模型和英文大小写，都从 `Data/prepared-clean` 的清洗后 token 流统计。换了训练数据再重建时，`bigram_counts.npz` 缓存要一起重算，不能用 `--from-counts`。

## 更新数据、训练模型

出厂模型按下面六步做出来，重做时照着走一遍。每一步都在数据目录里留下日志，删了多少可以查。

### 1. 清洗

训练文本和留出文本用同一套规则；`Training/test_cleaning.py` 逐条检查：该去掉的，不能误伤的正常打字，以及每条规则跑两遍和跑一遍结果相同。
- **个人信息换成“＊”**：规则只有一份，在 `Resources/redaction.json`（19 条，每条附说明）。训练数据清洗（`Training/cleaning.py` 的 `redact()`）和输入法的学习记录（`Engine/Redaction.swift`）都读这个文件，所以两边不会不一致。文件里的写法要让 Python 和 macOS 的正则（ICU）理解成同一个意思；`{{名字}}` 引用文件开头定义的公用片段，如网址结尾、域名后缀表、省份简称。规则必须跑几遍结果都一样：输入法每次启动都会把已存的记录重新打码一遍，否则会越删越多。
  - 规则覆盖的内容：
  - 邮箱；
  - 网址，包括：紧跟在汉字、数字或字母后的网址，语料里丢了“://”的网址（httpwww.bilibili.com…），以及 yys.163.com 这样的小写域名；
  - 6 位以上连续数字（电话、身份证、银行卡），分组写的卡号和手机号；
  - 账号：微信、QQ、抖音、快手、支付宝、淘宝、旺旺、钉钉、小红书、Line、Skype、Telegram 后面的账号。英文平台名后面要有“号”或冒号，只隔空格时账号里要有数字，所以 wxWidgets、“go on QQ immediately”不算；
  - “姓名、联系人、收件人”后面的名字，“@”后面的人（最多 20 个字，变成 @＊）；
  - 住址里的楼栋、单元和房号（11楼1211、3号楼2单元401、4603室），车牌；
  - 字母数字混排、20 位以上、数字多的乱码串：链接参数、校验码、图片文件名。数字只占一两位的名字保留，如 CanvasRenderingContext2D、RenaultMeganeRS275。
  - 日期、时间、价格、街道名、普通数字都保留。
- **去掉页面结构**：HTML 转义字符换回原字符；去掉以下内容，不论它们是单独一行，还是写在正文中间：
  - HTML 标签，包括语料里被截断的（“<img data-rawwidth正文…”“<a class”、行末的“</p”）；
  - 论坛排版代码（[size=14]）；
  - 论坛回帖抬头：“作者：… 回复日期：…”“@某人 3楼 2015-09-09 08:18:00”“某人 3楼 …”“发布者：… 发表日期：2013年05月21日”“来自：…客户端”；
  - 帖子签名、发布信息、封号通知、评论抬头，连同里面的用户名（“本贴由某某于…在〖…〗发表”“发布时间：… 编辑：…”“账号已被…封杀”）；
  - 精确到秒的时间戳（2017-04-18 23:01:00），以及跟着“上传、发布、来自…”的时间。只到分钟、写在正文里的时间保留（北京时间2011年1月2日8:30）。
- **去掉不是打出来的行**：8 字以上、数字和符号超过 60%、汉字和字母不到 30% 的行，例如表格、分隔线、时间戳。
- **整篇丢掉**：200 字以上、数字超过 20% 的文档，例如价目表、统计表。

页面结构那几条（标签、回帖抬头、时间戳）只用于训练数据，留在 `Training/cleaning.py` 里。

改规则时：
1. 先在 `test_cleaning.py` 里加用例；
2. 改 `Resources/redaction.json`（个人信息）或 `cleaning.py`（页面结构）；
3. 拿真实语料对比新旧规则，逐条看被改动的地方；
4. 用同一批句子比对 Python 和 Swift 两边的结果（应当完全一样），再跑 `Scripts/symtest.sh`。

```sh
P=Data/venv-train/bin/python
$P Training/prepare_data.py --out Data/prepared-clean --cased-from Data/prepared-clean > Data/prepared-clean/prepare.log   # 全部来源 → 清洗 → token 流、留出文本、验证集（多进程清洗约 20 多分钟；--workers 1 单进程约 2 小时，结果相同）
$P Training/make_words.py --out Data/prepared-clean                                # 词条行
$P Training/make_english.py --out Data/prepared-clean --data Data/prepared-clean   # 英文词行（含大小写）
(cd Training && python3 test_cleaning.py)                                          # 规则的用例
```

`--cased-from <目录>` 沿用该目录的 `vocab.json`（9026 个字符，含 26 个大写字母），换模型后词表不变；重新准备数据时指向现有的 `Data/prepared-clean`（`--out` 是同一个目录也可以，词表先读进来再写）。clean3 的数据清洗删去的字数：对话 0.05%、混合 1.26%、新闻 1.82%、论坛 11.72%、网页 1.08%、技术文档 1.64%，其余来源不到 0.1%。

### 2. 去重

```sh
(cd Training && ../Data/venv-train/bin/python dedup.py --data ../Data/prepared-clean)   # 跑两遍
```

`Training/dedup.py`（全量数据约 7 分钟、最多约 6 GB 内存，大部分是读进来的数据文件本身）：
- 整篇相同的文档只留一份；
- 20 字以上的行再次出现就删，短行不动（“哈哈哈”“好的”是真实打字）；
- 新闻、论坛、混合、网页、技术文档里，同一条 4 到 19 字的短行最多留 20 次，去掉“下载附件 保存到相册”“(小风)”“我也要打赏”这类模板。对话、触宝不限：这两类里出现得多的说法就是常打的说法；
- 重复内容留在质量高的来源里（人民日报 → 新闻 → 小说 → 对话 → 触宝 → 论坛 → 混合 → 网页）；
- 留出文本里有的内容，训练数据里一律去掉，测试句因此不会混进训练。

第二遍还会再删一些：第一遍删掉重复的长句后，有的文档变得完全相同。之后每再跑一遍，仍可能删掉极少一点（clean3 的数据跑第三遍只再删约 90 个字），可以忽略，跑两遍就够。训练用过的数据不要再原地去重：要试就在副本上跑，原样留着才能和训练时的 `SHA256SUMS` 对上。clean3 的数据两遍合计少掉的字数：论坛 21.6%、技术文档 7.1%、人民日报 6.9%、新闻 5.9%、网页 5.4%、对话 5.0%、触宝 2.2%、混合 0.6%（`dedup.log`、`dedup.first.log`）。

### 3. 按模型打分过滤

```sh
CURRENT=Data/clean3   # 当前出厂的模型
(cd Training && ../Data/venv-train/bin/python quality_filter.py --data ../Data/prepared-clean --model ../$CURRENT/model.pt score web mixed news forum)   # 约 5 小时
(cd Training && ../Data/venv-train/bin/python quality_filter.py --data ../Data/prepared-clean show web)       # 按分数从好到差抽样
(cd Training && ../Data/venv-train/bin/python quality_filter.py --data ../Data/prepared-clean filter web=3)   # 去掉最差的 3%，留出文本同一分数线，重写 val.bin
```

用当前出厂的模型（新模型要到第 5 步才训出来）给每篇前 256 字算平均损失。`filter` 会把分数文件跟着删掉的文档一起更新，分数和文档的个数对不上时报错停下，不会删错。分数线不能套公式：每类先 `show`，看分数最差那头的样本是什么，再定删不删、删多少。当前的数据：
- 网页去掉最差的 3%（42,262 篇，网页字数的 3.5%，留出文本 74 篇）：机器换词改写的文章、插了乱字的正文、字序打乱的页面、关键词拼接；
- 论坛、混合、新闻不删：分数最差的是古诗词、文言、百科词条、港式新闻，都是正常内容。

写得通顺的广告、推广按分数分不出来，照正常文本处理。

### 4. 重新生成判定集和学习检查集

```sh
$P Tools/make_fresh_eval.py                                                   # 对照第 1–3 步之后的训练数据查重
$E $C --eval Data/fresh/general_dialogue.tsv --no-rerank --dump-lattice Resources/general_lattices.tsv
cut -f1-3 Resources/general_words.tsv > /tmp/words.tsv && $E $C --eval /tmp/words.tsv --no-rerank --dump-lattice Resources/general_words.tsv   # 300 个单独的词
```

训练数据变了，测试句要重新查重，所以每次换数据都重新生成。两份学习检查文件是按词库排出的候选，换词库也要重新导出。当前这套：7 类 1,911 句，学习检查 473 句。

### 5. 训练

**模型**（`Training/model.py`）：字级 GPT，8 层、512 宽、8 个注意力头，窗口 64 个字，词表 9026（汉字、标点、26 个小写和 26 个大写字母），输入和输出共用一张字向量表，共 2990 万参数。

**目标**：给定前面的字，预测下一个字（交叉熵）。每一行训练数据是从某个来源的 token 流里随机截取的 65 个字。

**优化**（`Training/train.py`）：AdamW（β = 0.9 / 0.95，权重衰减 0.01），每批 256 行，梯度范数裁到 1.0。学习率先线性升温（前 1,000 步，不超过总步数的十分之一），再按余弦降到峰值的 5%。

**三段**（`Training/train_base.sh`）：

| 段 | 起点 | 步数 | 峰值学习率 | 配比 |
|---|---|---|---|---|
| 一 | 从头 | 80,000 | 6e-4 | 对话 .32、混合 .23、论坛 .16、新闻 .13、小说 .05、输入法真实语料 .05、人民日报 .03、词条 .03、英文词 .01 |
| 二 | 第一段 | 20,000 | 2e-4 | 对话 .28、混合 .18、论坛 .12、网页 .14、新闻 .09、输入法真实语料 .05、小说 .04、词条 .03、人民日报 .02、技术文档 .02、英文句子 .02、英文词 .01 |
| 三 | 第二段 | 6,000 | 5e-5 | 对话 .32、混合 .14、论坛 .10、网页 .09、词条 .08、新闻 .07、输入法真实语料 .05、英文词 .05、小说 .04、人民日报 .02、技术文档 .02、英文句子 .02 |

每一批按配比从各来源抽行。后两段接着前一段的权重训，优化器和学习率曲线重新开始。三段缺一不可：只用第三段的配比从头训，各项都更差。

**验证**：每 1,000 步用验证集（`val.bin`，各来源的留出文本）算一次平均损失，并存检查点；中断后重跑，从检查点接着训。验证集随训练数据一起变，不同数据训出的模型，验证损失不能直接比，判定看第 6 步。clean3 三段结束时的验证损失：3.198 → 3.136 → 3.140。

**数据**：训练用的数据必须和准备数据时写下的 `SHA256SUMS` 一致，`train_base.sh` 开训前逐个核对，有一个对不上就不训。

```sh
(cd Data/prepared-clean && ls *.bin vocab.json sources.json | xargs shasum -a 256 > SHA256SUMS)
Training/train_base.sh Data/prepared-clean Data/clean3
```

### 6. 导出与判定

```sh
cp Data/prepared-clean/vocab.json Data/clean3/vocab.json
LM_ROWS=32 Scripts/build_model.sh Data/clean3/model.pt build/clean3          # Core ML（32 行）
$P Training/export_weights.py Data/clean3/model.pt Data/clean3/weights        # GPU 权重
python3 Tools/extract_positions.py Data/clean3/weights build/clean3/positions.bin
```

先在临时目录里拼一份 `Resources/`：模型和位置表换成新的，其余链接到仓库里的文件。在那个目录里跑 `Scripts/gate.sh` 的各项，再和现在的模型逐句比较：

```sh
$E $C --eval Data/fresh/eval_dialogue.tsv --show-failures > new_dialogue.out   # 两个模型各跑一次
Tools/compare_failures.py old_dialogue.out new_dialogue.out                    # 新错、新对的句数和 p
```

- **判定只看判定集** `Data/fresh/eval_*.tsv`：句子来自清洗后的留出文本，训练数据里一条都没有，也从来没有用来调过参数。p 低于 0.05 才算真有差别，两次训练之间差几句是正常的。
- **防退步句子集**（`Data/eval_dialogue.tsv` 等）只用来防退步：引擎参数是在它们上面调的，有不少句子原样出现在训练数据里（对话 84/300，触宝 294/300），换模型时会偏向旧模型。它们的上文也按同样规则清洗过，目标句不受影响。
- **学习检查**：`Resources/general_lattices.tsv` 来自 `Data/fresh/general_dialogue.tsv`，和判定集来自不同的文档。输入法和 `--learn-demo` 用的是这同一份。

装机：把 `build/clean3/LanguageModel.mlmodelc` 和 `positions.bin` 拷进 `Resources/`，`Scripts/build.sh`，再跑一遍 `Scripts/gate.sh`。`weights.json` 只有结构参数，结构不变就不用动。

### 当前的数据和模型

- 数据：`Data/prepared-clean`。
- 模型：clean3，`Data/clean3`、`build/clean3`；8 层、512 宽、8 头、窗口 64 的字级 GPT，词表 9026，2990 万参数（`Training/model.py`）。换了出厂模型后，学习目录里的微调文件因模型指纹不符自动作废，再从打字记录重新学。
- 装机：模型 clean3（32 行导出）；词库与二元模型从 `Data/prepared-clean` 统计（`Data/lexicon`，命令和输出在 `Data/lexicon/rebuild-*.log`）。
- 流程要能重现：建库脚本重跑生成的词库与装机版逐行一致；`build_model.sh` 从同一个 `model.pt` 导出的 Core ML 权重逐字节一致；评测集生成脚本再生的文件与入库文件相同。

## 代码约定

- 引擎与 App 的接口只有 `CandidateDecoding`、`SentenceScorer` 和 `Learner`；App 不直接碰 SQLite 或 Metal。
- 不为具体词写规则；可调的量只在解码器的公开属性里，并能从 enginecli 的参数设置以便测量。
- 新按键行为先在 `Tools/symtest/main.swift` 加用例。
- 工具链注意：macOS 27 SDK 下 `@State` 是宏，在 SwiftPM 构建里从按钮闭包给 `@State` 赋值会报“self 不可变”，界面状态放进 `ObservableObject` 类。
