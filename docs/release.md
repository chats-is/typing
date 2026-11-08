# 发布

## 版本号

两个号都在 `Bundle/Info.plist`。按 App Store：只含数字和点，以数字开头、以数字结尾，最多三段，总长度不超过 18。

对外版本号是 `CFBundleShortVersionString`，三段 `主.次.修订`，如 `0.2.5`。

- 一个开发周期一个对外版本号。本地测试和公开发布用同一个号。
- 这个周期里不改。修改先留在工作区。
- 新的对外版本号必须大于上一个，不能重复。
- 公开发布后，下一轮修改开始时改成下一个。

构建号是 `CFBundleVersion`，一段到三段。不足三段时，缺的段当作 0。

- 所有对外版本合在一起计。构建号一直增大，用过的号不能再用。新的对外版本号不把构建号从 1 重新开始。
- 每次分发前加大，且大于以往用过的任何一个构建号。
- 发布按提交里的两个号原样打包，不再另加一号。

## 本地打包

对外版本号不动。打包前把构建号加 1，且大于以往用过的构建号。

```sh
Scripts/release/make_pkg.sh <对外版本号> dist dev
```

文件名是 `Typing-<对外版本号>-<构建号>.pkg`。安装包留在本机。不推送，不打标签。

## 发布

确认能用之后才发布。先在本地跑 `Scripts/gate.sh`，数字不低于 `docs/benchmarks.md`。CI 通过不算能用。

`.github/workflows/ci.yml`：推送到 `main` 或提 PR 时，在 `macos-26` 上构建、跑按键测试和四项词测试、打包 App。

`.github/workflows/release.yml`：推一个 `v*` 标签时发布。标签等于对外版本号，如 `0.2.4` 打 `v0.2.4`。对不上就停止。按提交里的 `Info.plist` 原样打包，不改版本号。跑 `Scripts/gate.sh`，然后产出：

- `Typing-<对外版本号>.pkg`：`Scripts/release/make_pkg.sh`。只允许 M 系列芯片、macOS 15 以上。装到 `/Library/Input Methods`。安装后为当前登录用户注册并启用。
- `Typing-<对外版本号>.dmg`：`Scripts/release/make_dmg.sh`。内含 `Typing.app`、指向 `/Library/Input Methods` 的链接和 `安装说明.txt`。
- `typing-resources-<日期>.tar.zst`：词库、二元模型、模型、权重、词表。给没有 Git LFS 的开发环境。

```sh
git tag -a v<对外版本号>
git push origin v<对外版本号>
```

发布不安装。

签名是本机临时签名，没有 Developer ID。正式分发用 `Developer ID Application` 签 App、`Developer ID Installer` 签 pkg，再 `xcrun notarytool submit` 公证，`stapler staple` 装订。

## 安装

1. 双击 `.pkg`；或打开 `.dmg`，把 `Typing.app` 拖进 Input Methods。
2. 输入法菜单里没有“智能输入”时，到 系统设置 › 键盘 › 输入法 添加，或注销后重新登录。

## 卸载

打开终端，执行：

```sh
"/Library/Input Methods/Typing.app/Contents/Resources/uninstall.sh"
```

这一条删掉安装的输入法和程序写入的学习记录。开发机上用 `Scripts/uninstall.sh`，删的是同样的内容。

## 发布前检查

- `Scripts/gate.sh` 全绿，数字不低于 `docs/benchmarks.md`。
- 更新 `docs/benchmarks.md`。规则有变化时更新 `docs/decisions.md`。
- 标签与 `README.md` 里的版本说明一致。
