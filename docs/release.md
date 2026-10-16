# 发布

## 自动发布（GitHub Actions）

- `.github/workflows/ci.yml`：推送到 `main` 或提 PR 时，在 macOS 15 运行器上构建、跑按键测试和四项词测试、打包 App。
- `.github/workflows/release.yml`：推一个 `v*` 标签（如 `v0.2`）时，把标签写进 `Info.plist` 的版本号（构建号取提交数），跑完整门槛 `Scripts/gate.sh`，然后发布：
  - `智能拼音-<版本>.zip`：`Typing.app`、`register`（注册工具）、`安装.command`、`卸载.command`、README、LICENSE，附 sha256；
  - `typing-resources-<日期>.tar.zst`：词库、二元模型、模型、权重、词表的归档，给没有 Git LFS 的开发环境。

```sh
git tag v0.2 && git push origin main v0.2
```

两个工作流都用 `actions/cache` 缓存 `.git/lfs`，LFS 的下载额度只在资源文件变化时消耗一次。

## 签名

当前是本机临时签名（ad hoc），没有 Developer ID。用户下载后 Gatekeeper 会拦截，`安装.command` 通过去掉隔离属性解决；要走正式分发需要在工作流里加入 Developer ID 证书和公证（`codesign --sign "Developer ID Application: …"`，`xcrun notarytool submit`），把 `Scripts/build.sh` 的签名步骤换成正式证书即可。

## 用户安装步骤

1. 解压，双击 `安装.command`。
2. 若输入法菜单里没有“智能拼音”，到 系统设置 › 键盘 › 输入法 添加，或注销后重新登录。
3. 卸载：双击 `卸载.command`。

## 发布前检查

- `Scripts/gate.sh` 全绿，数字不低于 `docs/benchmarks.md`。
- 更新 `docs/benchmarks.md` 和 `docs/decisions.md`（有新决定时）。
- 标签与 `README.md` 里的版本说明一致。
