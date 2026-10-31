# CPU 拳击训练的进攻奖励修复与验收

## 距离课程与快速进攻

新增训练开关 `--distance-curriculum`（需要 `--training-mode standing --official-rules`）。训练从骨盆水平间距 1.0 m 开始，依次为 1.4 m、1.8 m，最后恢复原训练环境的初始位置及扰动。仅改变重置时双方的水平位置，保持中点、朝向、高度及关节姿态；对手重新锚定站立目标。正式 Docker 场景不修改。

升级依据是最近 16 个完整回合：至少 12 局有学习方正式 HIT，最多 1 局出现学习方倒地或无效状态。仅 KO、辅助奖励、未完成回合及旧距离的在途回合不能作为新距离的升级证据。课程状态随检查点保存，续训保留阶段和已完成回合；仿真中的未完成回合重新开始。每局仍为 60 秒，未达标就停留当前难度。

课程自动启用 `--rapid-approach` 奖励配置，也可以在普通站立或自博弈训练单独使用该开关：

- 距离 ≥1.5 m 时，朝向对手的自身前进速度以 1 m/s 为训练目标；从慢速到目标速度获得逐渐增加的奖励，后退扣分，超速降低奖励。
- 0.9～1.5 m 之间逐渐减弱接近奖励；进入 0.9 m 后不再用此项鼓励向前冲撞。距离以双方骨盆水平距离定义，不是手部到目标的距离。
- 主动出拳奖励扣除自身躯干平移，使用朝目标方向的手部相对速度。在 0～2 m/s 范围持续奖励更快出拳，超过目标速度不继续加分；离目标越远奖励越小，挥向反方向不奖励。
- 降低持续朝向、动作变化和能耗项对进攻探索的压制：朝向权重 0.15，主动出拳 3.0，快速接近 2.0，动作变化 -0.02，能耗 -2e-6。完整参数见 `mujoco_rewards.py`，运行时写入 `environment.json`。

这些速度和权重是待训练验证的初始设置，不是模型已具备的能力。速度加快不保证达到 5 N·s，最终仍以真实接触冲量和官方 HIT 为准。不要放宽比赛裁判阈值。

```bash
python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" --policy-checkpoint "$START_POLICY" \
  --training-mode standing --official-rules --distance-curriculum \
  --num-envs 8 --updates 1024 --device cuda:0 --seed 43 \
  --save-interval 128 --output "$RUN_ROOT/distance-curriculum"
```

查看 `curriculum/stage`（0～3）、`curriculum/completed_bouts`、`ego/movement/approach_distance_m`、`ego/movement/approach_forward_speed_mps` 以及正式命中指标。完成课程后，用 `--policy-checkpoint` 开新自博弈阶段，去掉 `--distance-curriculum`，保留 `--rapid-approach`。评测入口默认不启用课程，始终检查原始距离，不因近距离训练命中就宣称比赛达标。

此功能不会自动启动新的长训练。已有权重需要用新配置训练后再录制和评测。

本版本针对已出现的“程序完成训练、策略仍没有有效得分”修复已确认的代码问题。模型权重不会因为代码更新自动改善；旧检查点必须重新评测，并按新奖励训练。正式命中阈值、冷却、KO 和比赛时间没有修改。

## 修复内容

1. `mujoco_rewards.terms(..., official_rules=True)` 明确排除旧的命中、被击中和高度终止代理项，由正式裁判计分替代。保留官方参考实现的 `active_hit_shaping_reward`（权重 1.5）、接近、朝向及出拳收拳辅助奖励。原先环境按 `hit` 子串过滤，误删了主动出拳信号。
2. `environment.json` 保存实际启用的奖励及权重。辅助奖励乘控制周期 0.02；正式分差及终局奖励不乘该系数。不要比较旧运行和新运行的总奖励来断定能力提升。
3. 新的球面策略自博弈阶段在没有指定 `--config` 时使用初始探索标准差 0.2，站立阶段为 0.1，与官方参考路线一致。显式配置优先，续训保留检查点配置。新阶段从已有策略继承网络尺寸，不再错误地回落到默认网络。
4. 增加正式裁判的接触诊断和 CPU 训练环境检查点评测。辅助奖励与正式得分分开记录，不将训练完成、接触或 KO 自动解释为学会有效出拳。

代码参考：[官方训练说明](https://github.com/HorizonRobotics/HoloMotion/blob/master/holoboxing/README.md)、[奖励配置](https://github.com/HorizonRobotics/HoloMotion/blob/master/holoboxing/training/envs/mujoco_rewards.py)、[奖励公式](https://github.com/HorizonRobotics/HoloMotion/blob/master/holoboxing/training/tasks/combat/mdp/rewards.py)。

## 先查看接触证据

`metrics.jsonl` 中学习方的键以 `ego/` 开头：

| 字段 | 含义 |
| --- | --- |
| `reward/active_hit_shaping_reward` | 是否得到主动出拳辅助反馈 |
| `referee/official_legal_contact_samples` | 合法手与合法目标的接触采样数，同一手同一物理时刻去重 |
| `referee/official_subthreshold_contact_samples` | 冲量未达阈值的采样数 |
| `referee/official_cooldown_contact_samples` | 因该手冷却而不计分的采样数 |
| `referee/official_contact_impulse_max_ns` | 每个控制帧中真实 20 ms 冲量峰值，再按采样批次取均值；不是整次更新的最大值 |
| `referee/official_hits` | 正式裁判本帧命中事件数 |
| `referee/official_score` | 正式累计比分；可能包含 KO 分，不等于命中次数 |

这些训练统计沿用现有日志的世界和时间平均方式，不能直接作为整场比赛命中总数。冲量指标来自裁判命中消耗积分前的记录，避免只查看帧末积分而遗漏已命中的接触。

接触采样为零时，先检查接近、距离与动作执行；有接触但冲量不足时，检查手部速度、方向和目标部位；有冷却拒绝时，不要误认为裁判失效。正式计分仍使用 20 ms 内 5 N·s、单手 200 ms 冷却。

## 在现有权重上建立新实验

使用已经安装的 CPU 训练环境和现有数据资源。不要覆盖旧输出目录，也不要恢复旧优化器及极低学习率来混淆修复效果。用 `--policy-checkpoint` 开始新阶段；其保留策略及归一化，重建优化器、价值网络、探索状态和对手池。

以下是短程诊断示例，路径变量需指向自己的有效资源，先选择空闲 GPU：

```bash
export HOLOBOXING_DATA=/absolute/path/to/prepared
export START_POLICY=/absolute/path/to/model_1024.pt
export RUN_ROOT=/absolute/path/to/new-experiment

python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint "$START_POLICY" \
  --training-mode standing --official-rules \
  --num-envs 8 --updates 32 --seed 43 --device cuda:0 \
  --save-interval 16 --output "$RUN_ROOT/standing"
```

32 次更新只是小规模检查点，不是行为验收标准。先用下面的评测观察命中和跌倒，确认进攻行为，再决定是否继续训练及转入自博弈。

```bash
python -m holoboxing.training.train_rl \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --amp-motions "$HOLOBOXING_DATA" \
  --policy-checkpoint "$RUN_ROOT/standing/model_32.pt" \
  --training-mode selfplay --official-rules \
  --opponent-pool-capacity 8 --opponent-refresh-updates 128 \
  --num-envs 8 --updates 128 --seed 43 --device cuda:0 \
  --save-interval 32 --output "$RUN_ROOT/selfplay"
```

此示例自博弈初始探索为 0.2；无需复制官方 128 环境的 GPU 物理配置。当前物理在 CPU 上串行步进，增加环境数应实测吞吐。更新次数不等于样本量：8 环境每次 24 步，128 次只有 24576 个学习方转换。

正常续训用 `--resume`，仍须显式保持 `--training-mode`、`--official-rules` 和对手池配置。代码或奖励改变后优先建立新实验，不应惯常通过 `--allow-resume-code-change` 绕过版本检查。

## 使用完整比赛验收

新增 `python -m holoboxing.training.evaluate_cpu --help` 支持加载现有策略并使用训练 CPU 后端和正式裁判。对固定策略对手应交换红蓝、使用多个种子；对站立目标的检查只能让学习方控制 A，报告会注明。

```bash
python -m holoboxing.training.evaluate_cpu \
  --assets "$HOLOBOXING_ASSETS" --lower "$HOLOBOXING_LOWER" \
  --checkpoint "$RUN_ROOT/standing/model_32.pt" \
  --standing-opponent --seeds 42 43 44 --device cuda:0 \
  --output "$RUN_ROOT/eval-standing"
```

对固定策略评测时，把 `--standing-opponent` 换为 `--opponent-checkpoint /absolute/path/to/frozen-opponent.pt`，程序自动交换红蓝。默认每局最多 3000 步（60 秒）；`--steps 4` 只能检查启动，不能通过完整比赛验收。`status=PASS` 仅表示执行成功；`scored_attack_gate_passed` 要求所有评测局完整、无无效状态，而且每局学习方至少产生一次正式 HIT。该门槛只证明本批次有有效出拳，不代表比赛能力已经充分验证。

先对旧权重和新权重使用完全相同的对手、种子、设备和配置，比较正式命中、比分、倒地和无效状态。未完成的短片段只能作为诊断；完整 0:0 或对方自行跌倒也不证明主动进攻有效。

评测保留训练环境的重置、扰动及观测噪声，属于训练后端行为检查，不是官方 Docker 资格评测。最终参赛仍需在官方环境中评测。

## 高层模仿学习仍然重要

当前只用 RoboStriker 训练动作先验的路线，没有完成官方高层模仿初始化。先验使动作目标落入较合理的表征空间，但不会直接教会策略根据对手状态选择出拳。

官方参考路线为动作先验 → InterHuman 配对实测数据上的 `train_sl` → 站立预热 → 自博弈。仓库已经提供 `holoboxing.training.train_sl`，其监督来源必须符合对应数据协议。不能通过把 RoboStriker 重命名为 InterHuman 或伪造对手观测跳过这一要求。

此次修复不会凭空生成配对数据或模仿检查点。若修复后的短实验仍没有进攻行为，应补充合法配对数据并执行模仿初始化，验证底层能跟踪目标，再扩大自博弈规模。动作重定向依赖单独环境，不要为此破坏已经可用的 CPU 训练环境。
