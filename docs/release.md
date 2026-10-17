# 发布

## 自动发布（GitHub Actions）

- `.github/workflows/ci.yml`：推送到 `main` 或提 PR 时，在 macOS 15 运行器上构建、跑按键测试和四项词测试、打包 App。
- `.github/workflows/release.yml`：推一个 `v*` 标签（如 `v0.2`）时，把标签写进 `Info.plist` 的版本号（构建号取提交数），跑完整门槛 `Scripts/gate.sh`，然后发布：
  - `Typing-<版本>.pkg`：标准 macOS 安装包（`Scripts/release/make_pkg.sh`，pkgbuild），装到 `/Library/Input Methods`，安装后脚本为当前登录用户注册并启用；
  - `Typing-<版本>.dmg`：磁盘映像（`Scripts/release/make_dmg.sh`，hdiutil），内含 `Typing.app`、指向 `/Library/Input Methods` 的链接和说明，拖放安装后到系统设置里添加；
  - `typing-resources-<日期>.tar.zst`：词库、二元模型、模型、权重、词表的归档，给没有 Git LFS 的开发环境。

```sh
git tag v0.2 && git push origin main v0.2
```

两个工作流都用 `actions/cache` 缓存 `.git/lfs`，LFS 的下载额度只在资源文件变化时消耗一次。

## 签名

当前是本机临时签名（ad hoc），没有 Developer ID。用户下载后 Gatekeeper 会拦截，要在 系统设置 › 隐私与安全性 里点“仍要打开”。要走正式分发需要两张证书：`Developer ID Application` 签 App（替换 `Scripts/build.sh` 的签名步骤），`Developer ID Installer` 签 pkg（`productsign`），再 `xcrun notarytool submit` 公证并 `stapler staple`。

## 用户安装步骤

1. 双击 `.pkg` 安装；或打开 `.dmg` 把 App 拖进 Input Methods。
2. macOS 27 起第三方输入法要先放行：系统设置 › 通用 › 登录项与扩展 › 扩展 › 输入方式，打开“智能输入”（安装脚本会自动打开这个面板；系统把放行名单存在受保护的 `com.apple.inputsources` 偏好里，安装程序写不了）。
3. 到 系统设置 › 键盘 › 输入法 添加“智能输入”；若列表里没有，注销后重新登录。
4. 卸载：删除 `/Library/Input Methods/Typing.app`，并在 系统设置 › 键盘 › 输入法 里移除“智能输入”；本机学习数据在 `~/Library/Containers/is.chats.typing`，不需要时一并删除。

## 发布前检查

- `Scripts/gate.sh` 全绿，数字不低于 `docs/benchmarks.md`。
- 更新 `docs/benchmarks.md` 和 `docs/decisions.md`（有新决定时）。
- 标签与 `README.md` 里的版本说明一致。
