# 架构

三条主线：输入、学习、设置。其余都是开发工具。

## 输入

每一键的路径（`App/TypingInputController.swift` → `Engine/InputSession.swift` → `Engine/LexiconDecoder.swift`）：

1. **格子解码（主线程，快）**。按键缓冲按音节切分（`SyllableMatcher`：全拼、简拼、未完成音节、邻键滑动），词库（`Lexicon`，SQLite，查询带锁和缓存）给出每一段能读成什么词，二元模型（`BigramModel`，词对概率，mmap）加上词频做 Viterbi，束宽 12，得到候选和整句读法。先按字面读，字面读不通时再做容错那一遍。增量缓存复用上一键的格子。
2. **模型重排（后台队列，慢）**。整句读法多于一个时，`DecodeResult.refine` 闭包在一个串行队列上调用语言模型（`SentenceScorer`）给每个读法打分，和格子分数合并后重排；落回主线程时若会话代数没变才采用。闭包在加锁时拍下模型、基准模型和学习过滤器的快照，不再读可变属性。中途来了新键，闭包开头就返回。结果没变化时返回 nil，不刷新界面。
3. **单词无上文**：按词库顺序（大众用法）；开启学习后，只对本人写过且没撤销的词加上“微调模型相对出厂模型的对数概率变化”，别的词不动。

语言模型是 8 层、512 宽的字级 GPT（`Training/model.py`），两种运行方式实现同一个 `SentenceScorer` 协议：

- `NeuralScorer`：Core ML，神经网络引擎，出厂模型和写回了学习结果的模型都走这里。
- `GraphScorer`：MPSGraph，GPU，直接读 `weights.bin/json`（fp16，可叠加微调张量），用于学习会话的校验，以及写回失败时的后备。

控制器之外的按键状态：中英文模式是进程级静态量（各应用一致），Caps Lock 状态变化和单独按下的修饰键按设置里的绑定触发功能；符号键直接上屏中文符号，面板提供英文替换，一个 0.25 s 的定时器在符号不再位于光标前时关掉面板。

## 学习

`App/Learning.swift` 调度，`Engine/Learner.swift` 执行，`Engine/LearningStore.swift` 存储（App 沙盒容器的 `Application Support/Typing/learning/`）。

- **记录**：每次上屏记一条（拼音、文字、前文），先压 3 秒；紧接着被删掉大半的不记。
- **触发**：键盘空闲 20 秒，且自上次会话以来新增 ≥ 20 条、间隔 ≥ 10 分钟（或新增 ≥ 120 条）。
- **会话**（后台队列，GPU）：按时间切出最近 10% 作为本人留出集；`GraphTrainer` 用 MPSGraph 从出厂权重开始，对最上两层和最后的归一化做 Adam 微调（每次都从出厂权重学当前全部记录，学到的东西永远是可见记录的函数）；学前、学后用同一运行方式（GPU）在三个集合上测首选率：本人留出集、大众句子集 `general_lattices.tsv`、常用词集 `general_words.tsv`；任何一项超过容差就丢弃。
- **暂停**：任何按键让会话在当前步、当前校验条之间停下，空闲 20 秒后继续；重置或关闭学习会取消会话，已开始的会话不再落盘、不再装回。
- **采纳**：`tuned.bin/json` 原子替换（先 .bin 后 .json）；`TunedModel` 把 26 个张量按名字写进出厂 `.mlmodelc` 的副本（`model.mil` 给出每个常量在 `weights/weight.bin` 的位置，条目头 64 字节：哨兵、类型、字节数、数据偏移），副本和 `GraphScorer` 比对几组对数概率一致后装入引擎，打字回到神经网络引擎；不一致则用 GPU 模型。
- **撤销**：设置里的标签把词写进排除名单，解码时的学习过滤器立刻生效；删记录后下次会话重新学。

所有图编译和运行经 `GraphLock` 串行（并发编译会在 MPSGraph 内部崩溃），编译只用 GPU（`optimizationLevel = .level0`，否则神经网络引擎规划会写几 GB 缓存）。

## 线程与锁

- 主线程：按键处理、快速解码、界面。
- `typing.refine` 串行队列：模型重排。
- `typing.learning` 串行队列：学习会话、模型写回与装载。
- `typing.learning.scan` 串行队列：设置窗口的“学到的变化”扫描，按键时让路。
- 锁序：解码器锁 → 词库锁；解码器锁 → 模型锁。`firstChoice` 在锁内解码、锁外跑模型。

## 文件格式

- 出厂权重：直接从 `LanguageModel.mlmodelc` 读（`model.mil` 给出每个常量的位置），加 `positions.bin`（完整 64 行位置表，fp16）和 `weights.json`（只含结构参数）；开发时的 `weights.json/.bin` 导出格式为 `{"config", "tensors":[{"name","shape","offset","count"}]}` + fp16 连续数据（微调文件为 fp32，`dtype_bits: 32`）。
- `bigram.bin`：`"BGM2"`、uint32 行数、uint32 词对数、每个首词一个起点偏移（uint32）、各行内排序的后词 uint32、float16 的 log2 P(w2|w1)；w1=0 表示句首。旧的 `"BGM1"`（uint64 键 + float32）仍能读。
- `lexicon.sqlite`：`lexicon(key, n, wid, word, logp, abbr)`、`english(key, word, logp)`，key 为撇号连接的音节。
- `entries.jsonl`、`state.json`、`exclusions.json`、`tuned.json/.bin`、`TunedModel.mlmodelc` + `TunedModel.stamp`：学习目录。
