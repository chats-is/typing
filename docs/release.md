# 发布

## 自动发布（GitHub Actions）

- `.github/workflows/ci.yml`：推送到 `main` 或提 PR 时，在 macOS 15 运行器上构建、跑按键测试和四项词测试、打包 App。
- `.github/workflows/release.yml`：推一个 `v*` 标签时，按提交里 `Info.plist` 的版本号和构建号原样发布（标签必须和对外版本号一致，如 0.2.4 打 `v0.2.4`，对不上就停止），跑完整门槛 `Scripts/gate.sh`，然后发布：
  - `Typing-<版本>.pkg`：标准 macOS 安装包（`Scripts/release/make_pkg.sh`，pkgbuild + productbuild），只允许装在 M 系列芯片、macOS 15 以上的 Mac 上，装到 `/Library/Input Methods`，安装后脚本为当前登录用户注册并启用；
  - `Typing-<版本>.dmg`：磁盘映像（`Scripts/release/make_dmg.sh`，hdiutil），内含 `Typing.app`、指向 `/Library/Input Methods` 的链接和说明，拖放安装后到系统设置里添加；
  - `typing-resources-<日期>.tar.zst`：词库、二元模型、模型、权重、词表的归档，给没有 Git LFS 的开发环境。

```sh
git tag -a v0.2.4 -m 0.2.4 && git push origin v0.2.4
```

两个工作流都用 `actions/cache` 缓存 `.git/lfs`，LFS 的下载额度只在资源文件变化时消耗一次。

## 签名

当前是本机临时签名（ad hoc），没有 Developer ID。下载后 Gatekeeper 会拦截，要在 系统设置 › 隐私与安全性 里点“仍要打开”。要走正式分发需要两张证书：`Developer ID Application` 签 App（替换 `Scripts/build.sh` 的签名步骤），`Developer ID Installer` 签 pkg（`productsign`），再 `xcrun notarytool submit` 公证并 `stapler staple`。

## 安装步骤

1. 双击 `.pkg` 安装；或打开 `.dmg` 把 App 拖进 Input Methods。
2. 若输入法菜单里没有“智能输入”，到 系统设置 › 键盘 › 输入法 添加，或注销后重新登录。
3. 卸载：运行 `Scripts/uninstall.sh`（停用、停进程、删除 `/Library/Input Methods/Typing.app`、清安装收据；加 `--data` 连 `~/Library/Containers/is.chats.inputmethod.Typing` 里的学习数据一起删）。手工卸载则删除该 App，并在 系统设置 › 键盘 › 输入法 里移除“智能输入”。

## 发布前检查

- `Scripts/gate.sh` 全绿，数字不低于 `docs/benchmarks.md`。
- 更新 `docs/benchmarks.md`，规则有变化时更新 `docs/decisions.md`。
- 标签与 `README.md` 里的版本说明一致。
