# HBot · G1 拳击仿真

本仓库用于团队参加 H-bot 拳击仿真比赛，围绕 Unitree G1 开发高层策略，
包括步法移动、转向接近、拳法执行，以及接触冲量和比赛行为评测。

高层策略通过五点目标驱动指定的冻结 HoloMotion V2 底层控制器。
五点分别为骨盆、左右手腕和左右脚踝；底层负责将目标转换为关节控制。
团队当前主要开发高层策略、训练环境与验证工具。

## 当前进展

截至 2026-11-04，仓库已整理以下可复用内容：

- 单机器人参考动作跟踪、技能训练与连续物理评测。
- 已学前进步法配合脚本控制的导航演示，完成过 5 米接近并停稳。
- `kangning-028` 转体参考，以及 `045` 短片段和其他连招的跟踪基线。
- 双机器人接触测量，按物理子步记录法向力、冲量和合法目标接触。
- 源码与数据来源校验、回归测试和 GitHub Actions 自动检查。

多方向快速转向、连贯接近并在拳距内停稳，以及稳定有效的混合拳攻击仍在开发中。
目前成果是局部仿真行为与评测工具，尚未完成整套比赛策略验收。

具体结果、限制和复现命令见[当前进展与基线说明](docs/boxing_validated_baseline.zh-CN.md)。

## 使用入口

| 需求 | 文档 |
| --- | --- |
| 了解已验证成果并复现实验 | [当前进展与基线说明](docs/boxing_validated_baseline.zh-CN.md) |
| 安装训练环境、了解本项目物理后端 | [规则对齐与独立评测工作流](docs/boxing_competition_workflow.zh-CN.md#cpu-physics-20261030) |
| 查阅比赛接口、模型导出与正式评测 | [拳击仿真使用文档](holoboxing/README.zh-CN.md) |
| 加入开发、提交改动和运行检查 | [协作约定](CONTRIBUTING.md) |
| 核对实验摘要与文件哈希 | [历史验证记录](experiments/validated_progress_20261103.json) |

环境与资产准备好后，按基线说明选择训练或评测入口，并为每次运行指定新的输出目录。
本地训练使用原生 MuJoCo CPU 物理，神经网络可使用 GPU；正式比赛评测环境单独管理。
上游教程中的训练配置与耗时不能直接视为本项目的验证结果。

## 主要目录

| 路径 | 用途 |
| --- | --- |
| `holoboxing/training/skills/` | 单机器人技能训练、参考动作、步法与导航演示 |
| `holoboxing/experiments/` | `045` 跟踪、双机器人冲量测量及来源记录 |
| `holoboxing/evaluation/` | 比赛评测、规则与接触语义 |
| `tests/` | 数据、控制接口和物理处理的回归测试 |
| `experiments/` | 纳入版本管理的实验摘要 |
| `docs/` | 环境说明、开发记录和复现文档 |

## 协作与验证

从最新 `main` 创建功能分支，通过 PR 提交改动。配置检查环境后运行：

```bash
python tools/check_boxing_baseline.py
```

涉及机器人行为的改动，还需提交指定冻结底层下的连续物理仿真结果。
单元测试通过或训练结束不等于动作、击打或导航验收通过。

模型权重、原始动作、视频和状态轨迹通过独立存储共享；仓库保留配置、来源和 SHA-256。
已验证前进步法的 checkpoint 尚未在 GitHub 发布，获取与兼容性说明见基线文档。

## 上游来源与许可

本项目基于 [HoloMotion / HoloBoxing](https://github.com/HorizonRobotics/HoloMotion)
代码与指定冻结控制器开展开发，并使用 [RoboStriker](https://github.com/yinkangning0124/RoboStriker-Preview)
的参考动作。上游项目是本仓库的技术与数据来源。

许可证和第三方声明见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)；相关源码中的版权说明继续保留。
