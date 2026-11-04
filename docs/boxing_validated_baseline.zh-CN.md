# G1 拳击：可复用基线与当前进展

更新日期：2026-11-04。此版本整理已验证的工具与局部行为，不是比赛策略验收版本。
所有执行都使用指定冻结 HoloMotion V2 底层；没有实机验证。

## 本次同步的内容

| 内容 | 保留原因 | 当前边界 |
| --- | --- | --- |
| `training/skills/` 单机器人参考条件 PPO、物理环境和评测 | 原始五个 teacher 源文件逐字节保留；已有前进步法 checkpoint | 不代表所有方向、拳法都学会 |
| `skills/waypoint.py` | 已完成 5 米前进并停稳的物理演示 | 转向是脚本请求；多方向快速转向尚未验收 |
| `skills/turn_reference.py` | 保留 `kangning-028` 原始转体参考、关节顺序和世界坐标一致性检查 | 参考动作加载正确不代表学会任意角度转体 |
| `experiments/combo45_*` | `045` 原参考跟踪与双机器人冲量测量可以复用 | 原参考不追踪对手，没有学得的瞄准或距离修正 |
| 接触几何分类 | 训练端与评测端共享准确的拳部、合法目标语义 | 保留现有裁判计分与诊断接口 |

保留 GitHub 原有的批量物理状态传输、线程选项及 FK CUDA graph 支持。
实验代码合入后通过 CPU 回归检查；原 GPU 报告来自 11 月 3 日运行，
不是对整理后代码重新运行整套 GPU 实验的宣称。

不纳入旧的 score-first、simple-strike、approach-combo、staged-upper-combo
等策略尝试及其训练配置，也不纳入尚未成功的导航 PPO manager 实现与权重。
原有 `main` 历史和已有功能保留；这些排除仅针对本次新增同步。
失败尝试仍在原本地工作目录中，不删除。

## 有效结果与未完成目标

机器可读的历史摘要与原文件 SHA-256 见
[`experiments/validated_progress_20261103.json`](../experiments/validated_progress_20261103.json)。
大体积视频、原始轨迹和权重不放入 Git。

- **前进到导航点**：目标 `(5, 0)` 米，11.6 秒结束；最终距离 0.280 米、
  平移速度 0.0140 m/s，连续站稳 1 秒，没有摔倒或中途重置。
  这是带脚本转向/停止逻辑的已学前进步法，不是已学会的通用导航策略。
- **拳距停止带**：目标 `(3, 1)`，要求距离 0.50–0.65 米；该次最终距离
  0.4976 米，触发过近停止条件，验收失败。该停止范围是骨盆 XY 间距，
  不等于拳头到目标表面的间距，也不能保证可出拳。
- **045 短参考跟踪**：3 段 × 参考/中立两种初始化，6 次完成且未摔倒。
  中立初始化时，动作区间腕部相对误差约 4.0–5.1 cm；肩肘仍有误差。
  此组原始协议有 2 秒末帧保持，不能把末帧保持理解为学得的收拳。
- **045 短片段冲量**：9 次双机器人测试中，1 次在原动作区间接触合法目标。
  最强连续接触持续 60 ms，法向冲量 7.143 N·s，20 ms 峰值 4.139 N·s，
  峰值法向力 303.95 N。全部未达到 20 ms 内 5 N·s 的必要阈值。
- **045 其他连招**：3 段，9 次实际双机器人 rollout 全部完成、没有摔倒；
  另有 5 种初始位置因机器人穿插而在步进前被拒绝，不能算作零冲量测试。
  移动连招仅有约 0.12 N·s 的轻微合法接触，另两段未命中合法目标。
  单机器人转体对照在 6 秒摔倒；与双机器人结果的差异原因尚未验证。
- **多方向快速导航**：被排除的导航 PPO 实验最终 17 项评测均未成功，
  主要停在 hold。连续快速转向、接近和拳距内停稳仍是后续目标。

未命中造成的零冲量不能说明机器人没有出拳力量。连续接触段也不自动等于一拳。
冲量结果只适用于这些参考、初始距离和对手姿态，不是最大力量估计或正式比赛得分。

## 环境与资产

1. 从 Git 仓库开发，参见 [协作约定](../CONTRIBUTING.md)。
2. 按 [独立拳击环境说明](../holoboxing/README.md) 安装训练依赖并取得比赛资产、
   指定冻结底层和 RoboStriker 原参考。沿用 MuJoCo 3.5.0，不混入其他实验环境。
3. `--assets` 指向含 `resources.json` 的比赛资产目录；`--lower` 指向含
   `config.yaml` 和 `exported/model_10500.onnx` 的指定冻结 bundle；
   `--source` 指向包含 `boxing_20251226/` 的 RoboStriker 根目录。
4. 视频需要 `imageio`、`imageio-ffmpeg` 及可用的 MuJoCo 渲染环境。
   无视频时传 `--no-video`。`045` 评测入口的真实底层执行使用 CUDA；
   CPU CI 不等价于这些 GPU 仿真测试。

下面使用 Linux shell 示例。从仓库根目录运行，把路径替换为自己的目录。
`HOLOBOXING_ASSETS`、`HOLOBOXING_LOWER` 也可以代替对应命令行参数。

```bash
export HOLOBOXING_ASSETS=/path/to/hbot-assets
export HOLOBOXING_LOWER=/path/to/holomotion-v2
export MUJOCO_GL=egl
SOURCE=/path/to/RoboStriker
```

### 前进 teacher 与导航演示

```bash
python -m holoboxing.training.skills.train \
  --source "$SOURCE" --skill locomotion --locomotion-mode forward \
  --mode train --num-envs 16 --updates 2048 --rollout-steps 32 \
  --save-every 256 --seed 55 --device cuda:0 --output results/forward-new

python -m holoboxing.training.skills.waypoint \
  --source "$SOURCE" --checkpoint /path/to/model_002048.pt \
  --goal 5 0 --hold-timeout 6 --output results/waypoint-new
```

已验证的旧 teacher SHA-256：
`46d78e9a65a6e74a798c370c1eac44edf80fcd06d9fa1b93ca017697d00f9dd3`。
该文件尚未在 GitHub 发布，需由项目维护者通过授权存储提供；也可按上面的配方重新训练，
但重训结果必须单独验收，不能假定与旧 checkpoint 完全相同。

旧 checkpoint 绑定原数据根路径等参考清单。跨机器更换路径时可能被严格兼容性检查拒绝；
不要跳过检查或直接改权重元数据，需另行做路径无关的来源等价验证。
`--catalog` 可指定原始目录清单文件，并严格校验哈希。仓库用 LF 固定相关源码与清单，
防止 Windows 换行转换改变哈希。

### 045 原参考与冲量评测

```bash
# 三个短片段，参考与中立初始化；原协议保留 2 秒末帧保持。
python -m holoboxing.experiments.combo45_reference \
  --source "$SOURCE" --output results/045-reference

# 三个短片段 × 三种初始距离，真实双机器人接触。
python -m holoboxing.experiments.combo45_impact \
  --source "$SOURCE" --output results/045-impact

# 其他三段长连招：单机器人对照 + 双机器人距离矩阵，无末帧保持。
python -m holoboxing.experiments.combo45_long \
  --source "$SOURCE" \
  --selection holoboxing/experiments/combo45_long_selection.json \
  --output results/045-long

# 只复测某段的指定初始距离；独立输出目录，不覆盖已有结果。
python -m holoboxing.experiments.combo45_impact_case \
  --source "$SOURCE" \
  --selection holoboxing/experiments/combo45_long_selection.json \
  --clip-id 1 --distances 0.65 0.70 0.775 --output results/045-turn-impact
```

长连招选段为 `[330,773)`、`[710,1137)`、`[1088,1475)`，50 Hz，
按原速度执行，进入时间 0.4 秒，无中途重置。原参考最后近似 T 字张臂，不能作为拳击架势收尾。
距离矩阵保留初始穿插的拒绝记录；`state=completed` 表示任务执行完毕，不等于动作或击打验收通过。

每个有效接触测试输出 `report.json`、`contacts-200hz.npz`、`contacts.jsonl`、
`states.npz`，录制时另有 `motion.mp4`。法向力按 200 Hz 的每个物理子步、每只手和目标类别
聚合后积分；报告分别统计进入段、原动作段和完整 rollout。
使用 `source_motion.records[].rolling_window_peak_ns` 判断原动作区间的 20 ms 峰值，
不要将进入段窗口残留或整段积分拿来比较 20 ms 阈值。

## 检查与后续工作

`python tools/check_boxing_baseline.py` 检查冻结源码哈希、Ruff、合并接口与 CPU 物理回归。
模拟底层的测试只验证接口、数据和物理处理，不证明真实冻结网络已执行成功。
GitHub Actions 自动运行同一检查；分支保护需另行配置。

下一步优先处理目标方向、动作相位和距离的协调，以及原参考末端的自然收拳/站稳。
在完成跨方向连续转向和接近的物理验收前，不将通用导航或 `045` 混合拳列为已完成。
