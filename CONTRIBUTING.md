# HBot 协作约定

当前可复用成果、复现方式及未完成目标见
[拳击基线说明](docs/boxing_validated_baseline.zh-CN.md)。

## 开发与提交

1. 从 GitHub 克隆仓库，基于最新 `main` 新建 `feat/<主题>` 或 `fix/<主题>` 分支。
   后续直接在 Git checkout 中开发，不在 ZIP 解压副本中长期积累改动。
2. 一个 PR 解决一个明确问题。提交和 PR 标题采用 `type(scope): summary`。
3. 提交前运行 `python tools/check_boxing_baseline.py`；涉及动作执行时，还要提供
   指定冻结底层下的连续物理仿真报告。不得把参考动画、mock 测试或离线损失当成技能验收。
4. 通过 PR 审核后合并，避免强制推送或整目录覆盖。建议仓库管理员为 `main`
   设置必须审核及 `cpu-checks` 状态检查；这份文件本身不启用 GitHub 分支保护。

## 模型、数据与实验

- 不提交权重、原始动作数据、视频、轨迹数组、个人服务器地址或凭据。
  通过经授权的独立存储共享大文件，仓库保存相对路径、获取说明、配置和 SHA-256。
- 记录代码提交号、工作区是否有改动、完整命令、随机种子、依赖版本、底层模型哈希和数据哈希。
  新的 `045` 评测入口会记录 Git 身份和关键源码哈希。
- 每次使用新的输出目录；验收报告保留失败与被拒绝的案例，不能只统计成功样本。
- `holoboxing/experiments/frozen_skill_sources.json` 固定原始五个 teacher 文件。
  当前同步保持其原始字节不变。后续确需修改时，另开 PR，明确旧 checkpoint 兼容性，
  同步更新哈希并重新验证；不要仅更新哈希来绕过检查。
- 未成功的策略尝试不必进入主分支。可在本地保留，并在成果文档中简要记录未完成的目标。

## 本地检查环境

使用 Python 3.11 的独立环境；下面只安装 CPU 检查依赖，实际训练环境见项目 README。

```bash
python -m venv .venv-checks
# 激活环境后执行；Windows 使用 .venv-checks\Scripts\Activate.ps1。
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r environments/requirements_boxing_checks.txt
python tools/check_boxing_baseline.py
```

CI 不下载冻结模型和数据、不执行实机控制；CUDA 专用检查在 CPU 环境会明确跳过。
