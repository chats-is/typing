# 智能输入法

智能输入法：macOS 原生中文输入法，系统输入法菜单里显示为“智能输入”。现在是拼音输入，下一步在同一个输入法里加 AI 语音输入功能。拼音部分：词库 + 语言模型决定候选，模型只在本机、用你确认过的文字继续学习。不联网，不记录你的选择次数，没有针对具体词的手写规则。

- **输入**：全拼、混拼（全拼和简拼混用）、简拼；按键滑动容错；中英混合（大写字母原样保留，英文单词作候选，有中文候选时中文优先）；标点键出中文符号（和系统拼音一样），英文符号在英文模式下打；Caps Lock 或 Shift 切换中英文（设置里可改）。
- **学习**：记录你确认上屏的文字和拼音（写入前去掉邮箱、网址、手机号和证件号这类长数字，密码和验证码提示后的输入不记；英文只记单词，你打过的英文单词词表里没有也会成为候选），键盘空闲时在 GPU 上微调语言模型的最上两层，训练前后用你最近的输入、一组大众句子和一组常用词校验，变差就回滚；采纳后把学到的参数写回 Core ML 模型，打字仍在神经网络引擎上跑。学到的内容在设置里以标签列出，可逐个撤销、删记录、整体重置或关闭。
- **设置**：一个页面：中英文切换的按键；自动学习的开关、状态、立即训练和重置；“改变的首选词”和“训练用的输入”两个标签列表，可搜索、分页，逐个删除；关于。

## 结构

```
App/        输入法本身：IMKInputController、候选面板、设置窗口、快捷键、学习调度
Engine/     引擎：词库（SQLite）、二元模型、音节匹配、格子解码与模型重排、输入会话；
            学习：记录库、MPSGraph 训练器、Core ML / MPSGraph 两种模型运行方式、学习会话、写回模型
Resources/  装进 App 的资源：lexicon.sqlite、bigram.bin、LanguageModel.mlmodelc、positions.bin、weights.json、vocab.json、
            general_lattices.tsv、general_words.tsv（大文件不在仓库里，见下）
Bundle/     Info.plist、entitlements、本地化
Scripts/    build.sh install.sh uninstall.sh gate.sh symtest.sh build_model.sh
Tools/      开发工具：enginecli（评测与模拟）、symtest（按键测试）、syspinyin（系统拼音对照）、
            词库与评测集的生成脚本（Python）
Training/   语言模型的数据准备、训练、导出（PyTorch → Core ML / fp16 权重）
Data/       评测集（只保留主测试用的几份；语料、检查点、生成表都不入库）
docs/       架构、开发流程、决策记录、基准数字
```

## 构建与安装

只支持 M 系列芯片的 Mac，macOS 15 以上；构建需要 Xcode 命令行工具。

```sh
Scripts/install.sh      # swift build → build/Typing.app → /Library/Input Methods（要一次管理员密码），并登记
Scripts/uninstall.sh    # 停用、停进程、删除 /Library/Input Methods/Typing.app、清安装收据；加 --data 连学习数据一起删
```

`Resources/` 里的三个大文件（lexicon.sqlite 109 MB、bigram.bin 98 MB、LanguageModel.mlmodelc 57 MB）用 Git LFS 存放：装了 `git-lfs`（`brew install git-lfs && git lfs install`）再 clone 就能直接构建。没有 LFS 的环境可以用发布包：`Scripts/package_resources.sh` 打包带校验和的归档，`Scripts/fetch_resources.sh <归档>` 解到 `Resources/`。也可以按 `docs/development.md` 用工具链重新生成。`Scripts/build.sh` 在缺文件时会报错指出。

## 测试

```sh
Scripts/gate.sh         # 装机前的全部门槛：四项词测试、按键测试、并发压力、节奏模拟
```

规矩是“改一处、测全部、不降才装机”。数字见 `docs/benchmarks.md`。

## 发布

推一个 `v*` 标签，GitHub Actions 会跑完整门槛并发布 `Typing-<版本>.pkg`（标准安装包）、`.dmg`（拖放安装）和资源归档。细节与签名说明见 `docs/release.md`。

## 文档

- `docs/architecture.md` 引擎与学习的数据流、线程与锁、模型文件格式
- `docs/development.md` 日常开发、评测工具、词库与模型的重建流程
- `docs/decisions.md` 已经定下、不再反复的产品与技术决定
- `docs/benchmarks.md` 基准数字与测量方法
- `docs/release.md` 自动发布、签名、用户安装

## 许可

GPL-3.0，见 `LICENSE`。
