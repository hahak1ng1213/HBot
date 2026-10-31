# 拳击规则对齐与独立评测工作流

## 2026-10-31：node04 真实采集与首次 PPO 更新修复

CPU 数据采集器现在将场景顺序的关节位置、速度按名称转换为 ONNX/训练物理的
规范顺序，控制器输出直接按该规范顺序写入物理目标。控制器关节一致性检查保留。
真实 RoboStriker 采集完成 6 个非测试动作文件、4858 条样本，并通过内置数据校验。

球面策略的赛况归一化常量改为预先注册的非持久 buffer，避免在 CUDA 图捕获中
从 Python 列表创建 GPU 张量；既有检查点键和冻结先验不变。源码 ZIP 没有 Git
元数据时仍记录 revision=null，但不再把预期的 Git 查询失败打印为 fatal。
node04 的 PyTorch 2.6.0 环境中，CUDA 捕获/重放/梯度一致性及球面 PPO、模仿导出
相关测试共 12 项通过。

修复后已完成 8 个环境、32 次 standing PPO 更新（GPU 0，原生 CPU MuJoCo），
训练循环约 83.46 秒、含初始化与保存总计约 127.94 秒。结果保存在
`/home/zhanghan/runs/hbot/robot-standing-v2`，末次检查点 `model_32.pt`；
日志为同级 `robot-standing-v2.log`。这证明采样、CUDA 图、优化和保存链路完成，
不代表已验证站立稳定性、比赛得分或胜率。原 `robot-standing` 是失败诊断目录。

本次增加的是可测试的研究入口。CPU 裁判和评测编排测试不构成 GPU 物理验证或胜率证据；资源、训练依赖、上层检查点全部就绪后才能产生基线和消融结果。低层控制器及正式评测代码的计分规则保持不变。

## 训练热路径校验精简

CPU 后端之后的本次精简只涉及拳击训练控制链路，未改动正式裁判或训练算法。
删除中间层重复的位姿形状/有限值检查、刚归一化后的四元数单位长度复查、
初始化已确定的 ONNX 输出结构重复检查，以及控制转换前后的物理步计数断言。
异常环境日志直接记录已检测出的世界编号，不再重复遍历张量并故意抛出/捕获异常。
正常观测每步检查一次，仅重置后的异常行追加复查。

仍保留外部输入接口、关节顺序、模型身份和数据划分检查；低层循环缓存、
送往物理引擎的电机目标、物理求解器和训练数值异常检查也保留。
文件路径边界、输出覆盖限制、正式计分与检查点兼容性检查没有删除。
当前没有做吞吐基准，不能把减少检查等同于已测得的加速。

精简后核心回归 **64 项通过**，包括新增的异常目标拒绝、异常观测恢复及
重置失败中止测试。日志：`validation-cleanup-tests.log`。
更早的扩展回归记录和真实资产/模型未验证的限制仍见下文。

<a id="cpu-physics-20261030"></a>
## 2026-10-30：原生 MuJoCo CPU 训练后端

本分支的拳击 RL 和 measured rollout 数据采集已经替换为原生 MuJoCo CPU 物理。
训练环境不再导入 `warp`、`mujoco_warp`，也不再调用其驱动检查。旧物理、环境、
ONNX 和采集实现留档于仓库根目录 `.cpu-backend-original/*.txt`，不参与 Python 导入。
神经网络、优化器和冻结低层 ONNX 的运行设备由 `--device` 选择，默认仍为 `cuda:0`；
`--device cpu` 可用于纯 CPU 验证。高层 1097→225 接口、冻结低层权重、PPO/AMP、奖励和正式裁判规则保留。

每个并行世界拥有独立的原生 MuJoCo 模型和状态，避免 COM 随机化互相污染。
物理步长仍为 5 ms，控制周期仍为 20 ms；每个子步用 `mj_contactForce` 读取实际求解器接触力，
转换到世界坐标后交给已有裁判。观测 FK 使用独立数据，不覆盖本子步的接触证据。
当前 CPU 世界串行步进，尚未实现多线程物理加速。

### 安装与启动

在 Linux 仓库根目录创建 Python 3.11 独立环境。CPU 物理不需要 CUDA toolkit 或 Warp；
若神经网络使用 GPU，仍须安装与显卡、驱动匹配的 PyTorch 和 ONNX Runtime GPU。
1080 Ti 的候选安装配置如下（不代表在目标服务器验证通过）：

```bash
python3.11 -m venv .venv-holoboxing
source .venv-holoboxing/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r holoboxing/training/requirements.txt
python -m pip check
```

纯 CPU 环境改用 PyTorch CPU wheel 和 `holoboxing/training/requirements-cpu.txt`，
不要在同一环境同时安装 `onnxruntime` 和 `onnxruntime-gpu`。

资源准备命令仍见下文。资源齐全后，先运行：

```bash
python -m holoboxing.training.preflight \
  --assets thirdparties/HoloMotion_assets/holoboxing --lower models/holomotion-v2 \
  --device cuda:0 --output results/preflight-cpu-physics.json
python -m holoboxing.training.runtime_probe \
  --assets thirdparties/HoloMotion_assets/holoboxing --lower models/holomotion-v2 \
  --device cuda:0 --output results/runtime-cpu-physics --num-envs 1 --steps 4
```

这里 `cuda:0` 仅指定神经网络设备，物理始终在 CPU。纯 CPU 检查将其换成 `cpu`。
数据采集入口 `python -m holoboxing.data prepare ... --device cuda:0` 同样如此；
省略数据采集设备时自动选择 CUDA（不可用时用 CPU）。

使用下文 RoboStriker 路线准备数据、训练先验，再执行站立预热和自博弈。
`train_rl` 未显式指定配置、环境数或恢复训练时，默认使用 8 个世界；已有显式配置仍会被尊重，
不要直接套用 4096 环境的旧 GPU 配方。建议所有启动命令明确传入 `--num-envs 8`。

### 数据、恢复和评测边界

CPU 采集数据带独立 recipe `measured_pair_short_window_cpu_v2`，不能把旧 Warp 采集数据
冒充为本后端数据；原始 RoboStriker 动作仍可复用，measured rollout 需要重新生成。
RL 身份记录包含 `physics_backend=mujoco_cpu`，旧 Warp 训练检查点不能作为同一运行直接 `--resume`；
可按现有 `--policy-checkpoint` 新阶段初始化规则复用兼容策略，重新评估效果。

官方 Docker 评测、原生对手专项评测 `evaluate_native`、可选 SMPL/InterHuman 的 Newton 重定向
仍保留各自原有依赖。这些是独立工具，不在 RoboStriker→CPU 数据采集→先验→RL→导出路径中。
训练后应在官方环境评测最终 ONNX；CPU 和定制 Warp 物理不承诺轨迹、吞吐或比赛分数一致。

### 本地验证范围

新增 `tests/test_boxing_cpu_physics.py` 使用真实 MuJoCo CPU 步进和合成 MJCF，
覆盖原生步进对照、力矩限幅、独立 COM 随机化、局部重置、速度/Jacobian 对照、
接触力/裁判映射、无 Warp 导入、ONNX CPU I/O、环境控制奖励和 measured rollout 采集。
环境与采集集成测试使用替身低层控制器，不代表真实冻结模型或完整比赛验收。
本机测试环境是 Python 3.10 / Torch 2.7.1 CPU / MuJoCo 3.5.0 / ORT 1.22.0；
不是上面的 Linux Python 3.11 GPU 环境。本地缺少赛事资产和冻结模型，未完成真实赛事场景训练。

2026-10-30 实际结果：新增 CPU 后端 11 项测试通过；最终核心组合回归 61 项通过。
扩展拳击回归为 105 项通过、1 项跳过、4 项失败；4 项失败均位于本次未修改的
InterHuman 下载测试，Windows 下报 `Data path escapes its root: annots/...`。
这些失败没有通过删除测试或放宽路径检查掩盖。
另行尝试的旧 HoloRetarget H5 测试因其引用 Warp 运动跟踪模块而无法收集；
本后端新增的纯 H5 写入/读取往返测试已通过。完整日志为仓库根目录
`cpu-backend-tests.log` 和 `cpu-backend-extended-tests.log`。
HDF5 写入模块的 SMPL/Warp 导入已改为仅在调用可选重定向函数时发生；
独立环境测试显式禁止导入 Warp/Newton，验证拳击训练及数据写入入口。

下方 2026-10-29 的验收记录是替换前历史记录，不适用于本次新后端。

## 2026-10-29 实际验收状态

- 本机和服务器已补齐固定低层模型、49 项拳击资源，以及 RoboStriker 前 4 个原始动作组及镜像（8 个文件），逐文件通过原项目清单校验。
- 验证机器使用独立目录 `~/projects/holoboxing-review-20261029`，解释器为其中的 `.conda-boxing/bin/python`。环境准备未改动已有 Conda 环境或系统驱动。
- Python 3.11，PyTorch 2.9.1+cu126、MuJoCo 3.5.0、MuJoCo-Warp 3.5.0、Warp 1.11.0、ONNX Runtime GPU 1.22.0；`pip check` 通过。安装清单保存于 `environment-lock.txt`。CUDA 12.6 的 PyTorch 在 GTX 1080 Ti 上执行实际 CUDA 张量运算通过。
- 46 项 CPU 回归测试在上述固定版本环境中通过，覆盖共享裁判、接触映射、分组隔离、PPO、阶段恢复、外部对手保留、观测消融和 ONNX 导出数值一致性。测试使用人工小样本，不构成训练成绩。
- 公开动作的 reference-only 数据已准备在 `results/robostriker-reference-pinned`，包含相互隔离的 train / validation / test 原始动作组，分别有 3354 / 1504 / 1308 个参考窗口。没有取得或使用授权 SMPL / InterHuman 文件，没有启动正式策略训练。
- **GPU 物理测试失败，未完成任何仿真训练或比赛基线。** `results/runtime-probe/report.json` 记录：驱动 535.230.02（CUDA driver API 12.2），MuJoCo-Warp 在 `check_toolkit_driver()` 拒绝启动，错误为 `Minimum supported CUDA version: 12.4.`，缺少 CUDA conditional graph 支持。这发生在创建物理模型时，尚未执行训练环境的接触采样。

因此，当前没有经过实战验证的最佳检查点、胜率、消融收益或参赛资格结论。需要管理员升级到支持所需 CUDA 图功能且兼容 GTX 1080 Ti 的驱动，或切换到符合要求的机器，再重复运行时检查。仅重装 PyTorch / CUDA 用户态依赖不能解除已测得的驱动功能限制。NVIDIA 的 [CUDA 12.4 发布说明](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/)列出 Linux 工具包对应驱动 550.54.14；该版本号是排查依据，升级后仍须实测整条固定依赖链。

完整回归命令：

```bash
PYTHONPATH=.:tests .conda-boxing/bin/python -m pytest -q \
  tests/test_boxing_competition_workflow.py tests/test_boxing_information_contract.py \
  tests/test_boxing_training_stages.py tests/test_boxing_spherical_ppo.py \
  tests/test_boxing_spherical_imitation.py tests/test_boxing_training_splits.py \
  tests/test_boxing_export_parity.py
```

## 资源准备与预检

若服务器连接 Hugging Face 超时，可在下载前设置 `export HF_ENDPOINT=https://hf-mirror.com`。
比赛资源 bootstrap 和冻结底层模型 download 均支持该变量；未设置时仍使用官方地址。
下载保留固定 revision、大小及 SHA256 校验，客户端请求标识为 `HoloBoxing/1.0`。

首次提交的最短路径是：换到兼容机器 → 收集真实低层 rollout 训练数据 → 训练动作先验 → 单种子站立预热与自博弈 → 导出高层 ONNX → 用当前赛事镜像做一次接口／时延验收并查看完整对局 → 上传 ONNX。现有 reference-only 窗口不能代替 rollout 数据；当前还没有可提交的已训练上层模型。

首次提交可采用公开 RoboStriker 的先验直接 RL 路线，暂不依赖 SMPL / InterHuman，也不必等待完整多种子消融和大规模对手池。后续再以独立对手成绩改进模型。

按后续工作约定，已去掉评测编排中每局重复计算模型 hash，以及 normalizer 每次前向新增的赛况有限值／形状检查；保留入口契约、检查点兼容性及最终导出验收。评测期间应保持输入模型文件不变。本次精简未重新执行回归或 GPU 验证，上述 46 项通过是精简前的验证记录。后续对固定候选集中做一次提交验收，模型或环境有实质变化时再复测。

在仓库根目录、教程指定的 Linux Python 3.11 拳击环境中执行：

```bash
python -m holoboxing.training.bootstrap_resources --assets thirdparties/HoloMotion_assets/holoboxing
python -m holoboxing download --directory models/holomotion-v2
python -m holoboxing.training.preflight --assets thirdparties/HoloMotion_assets/holoboxing --lower models/holomotion-v2 --output results/preflight.json
```

资产下载锁定公共仓库 revision，逐文件验证大小和 SHA256，写入 download-receipt.json。现存不同内容的文件不会被覆盖。控制器继续使用原项目带校验的断点下载。不要把 Windows 源码归档中的 `holoboxing/assets` 路径文本当作实际资源目录；显式传入上述 assets 路径。

preflight 记录资源校验、解释器、模块可用性和显卡驱动探测；返回 2 表示条件不足。即使返回 0，也只表示可以继续做运行时探测，不表示依赖版本、CUDA provider 或比赛已经验收。

依赖就绪后运行有界检查；它执行中性参考与固定低层，不训练上层，也不是比赛基线：

```bash
python -m holoboxing.training.runtime_probe \
  --assets thirdparties/HoloMotion_assets/holoboxing --lower models/holomotion-v2 \
  --output results/runtime-probe-new --num-envs 1 --steps 4
```

## 无 SMPL 时的公开数据基线路线

新增 `--prior-normalization-dataset robostriker` 只用于 `--prior-checkpoint` 的先验直接 RL 初始化：用真实低层 rollout 收集的 RoboStriker **训练组**拟合观测统计，跳过监督模仿。SL 默认仍严格选择 InterHuman。这条路线与教程的 InterHuman+SL 条件不同，应单独报告，不能复用论文的表现结论。

以下命令是驱动修复后才可执行的配方，不是已完成的实验记录。保持原始组及镜像同分区；两次准备采用相同 seed 和划分比例。

```bash
export HOLOBOXING_ASSETS="$PWD/thirdparties/HoloMotion_assets/holoboxing"
export HOLOBOXING_LOWER="$PWD/models/holomotion-v2"
PY="$PWD/.conda-boxing/bin/python"

"$PY" -m holoboxing.data download robostriker \
  --output holoboxing/data/raw/RoboStriker --limit-groups 4 --workers 2
"$PY" -m holoboxing.data prepare --source holoboxing/data/raw/RoboStriker \
  --output holoboxing/data/processed/robot-baseline --assets "$HOLOBOXING_ASSETS" \
  --seed 42 --validation-fraction 0.1 --test-fraction 0.1 --workers 1
"$PY" -m holoboxing.data verify \
  --output holoboxing/data/processed/robot-baseline --assets "$HOLOBOXING_ASSETS"
"$PY" -m holoboxing.training.prepare_prior --source holoboxing/data/raw/RoboStriker \
  --assets "$HOLOBOXING_ASSETS" --output results/robot-reference \
  --seed 42 --validation-fraction 0.1 --test-fraction 0.1
"$PY" -m holoboxing.training.train_prior --data results/robot-reference \
  --output results/robot-prior --hidden-dims 512 256 --batch-size 1024 \
  --epochs 1000 --seed 43 --device cuda:0
"$PY" -m holoboxing.training.train_rl \
  --amp-motions holoboxing/data/processed/robot-baseline/prepared \
  --prior-checkpoint results/robot-prior/best.pt --prior-normalization-dataset robostriker \
  --official-rules --observation-mode current_state --training-mode standing \
  --num-envs 8 --updates 32 --save-interval 16 --seed 43 --output results/robot-standing
"$PY" -m holoboxing.training.train_rl \
  --amp-motions holoboxing/data/processed/robot-baseline/prepared \
  --policy-checkpoint results/robot-standing/model_32.pt \
  --official-rules --training-mode selfplay --num-envs 8 --updates 384 \
  --opponent-refresh-updates 128 --save-interval 64 --seed 43 --output results/robot-selfplay
```

这是小规模可复现性起点，环境数和预算尚未在这台机器上验收。保持网络宽度、学习率和奖励辅助项原设置，先验证吞吐与稳定性，再进行多种子实验。384 次更新覆盖多个对手刷新点；默认每次 24 步时，两次刷新间有 `128 × 24 × 0.02 = 61.44 s` 的模拟时间，可容纳完整回合。对手池不会凭这些设置自动产生多种打法，仍需导入不同种子／风格的冻结检查点。

## 正式规则训练模式

现有 `train_rl` 命令增加 `--official-rules`。保留不带此参数的旧训练行为以供复现对照；新的竞赛实验应显式启用它。

- 每个 5 ms 物理子步保留实际接触证据，仅正法向力参与冲量累计，不使用摩擦力替代。
- 每个并行世界调用同一个正式 `MatchStateMachine`，直接复用合法命中、双手独立冷却、冲量消耗、KO、同时倒地和 60 秒超时规则。
- 移除该模式下旧的二值命中／被击和低高度代理奖励，加入己方减对方的正式得分增量及终局胜负奖励，保留其他辅助任务项及 AMP。
- 取消该模式的累计接触 5 秒提前终止；比赛超时视为真实有限时域终局，不再做 time-limit bootstrap。
- 每次回合重置分数、时钟、冲量与冷却状态；正式模式不随机缩短初次回合。

当前适配器是正确性优先的 CPU 裁判桥接，每子步有 GPU→CPU 同步；不要直接按 4096 环境配置启动。先进行少量环境的接触证据对照、KO 边界及吞吐检查，再决定是否需要实现并验证向量化裁判。该模式仍保留原训练的扰动／随机 push，它们用于训练分布；最终成绩必须来自未修改的正式评测。

## 时间分差与历史消融

新课程阶段可使用：

| 对照 | 新增选项 |
| --- | --- |
| 当前状态基线 | `--official-rules --observation-mode current_state` |
| 当前状态加赛况 | `--official-rules --observation-mode current_state --match-context` |
| 自身历史加赛况 | `--official-rules --observation-mode history10 --match-context` |

赛况从训练裁判进入 obs 最后两维，贯通 normalizer、球面策略和 ONNX 导出。时间除以 60、分差除以 1000 后，经零初始化的可训练 2→32 投影加入 latent 输出。旧策略不启用赛况时保持原输出语义。历史方案使用现有自身十帧历史，不增加对手速度或额外输入维度。

选择不同观测方案必须建立新阶段，不允许在 `--resume` 时切换；历史 checkpoint 用 `--policy-checkpoint` 或 SL checkpoint 初始化。旧检查点若要恢复完整优化器状态，应使用原版本代码；新参数不能冒充旧优化器的连续恢复。

同一初始模型、同一数据、相同预算和多个配对训练种子运行三组，保存各自的检查点与导出模型。以下评测矩阵负责比较结果；尚未产生数据前不能声称任一信息方案提高胜率。

## 外部冻结对手

selfplay 新增可重复参数 `--opponent-checkpoint PATH`，可导入不同种子／打法的可信 RL 检查点。导入校验资产、观测／解码器契约、参数形状、有限值、统计量和内容 hash，拒绝重复权重。网络形状与表示契约必须兼容；这不是任意原生控制器插件。

外部对手随快照和归一化统计保存在池中，刷新淘汰时保留；至少预留两个槽给初始 learner 和近期 learner。外部对手和历史对手均参与均匀抽样。池状态与随机状态一起保存／恢复。尚未实现 Elo/PFSP 难度采样，不能把多种子导入称为完整联盟。

## 独立评测与选模

复制并填写 `holoboxing/training/configs/evaluation_matrix.example.json`。每个候选填写真实训练种子、观测方案、已导出的 ONNX 和训练对手的模型 hash；示例中的空列表不是来源声明的替代。对手选择未参与训练的数据集合。路径相对 plan 文件解析。

```bash
python -m holoboxing.evaluation.matrix --plan path/to/validation-plan.json --output results/validation-matrix
```

每个候选与每个对手各进行红蓝两局，不生成无意义的确定性重复。保存计划、模型 hash、逐局 summary／失败原因和回放路径。任何失败或未结束对局都会使该候选失去本次选模资格，而不是被悄悄丢弃。验证集按胜=1、平=0.5、负=0 的均值选择检查点，分数与分差供诊断；这个分值不是官网 Elo。

训练可用 `--save-interval N` 定期保存候选；先用现有 `holoboxing.training.export` 将每个候选导出为 ONNX。汇总包含有效命中、被击、KO 与被 KO 数。指定 `baseline_variant` 后，报告按相同训练种子计算方案间 match points 差值，标记缺失／失败种子；每个方案／种子只能放一个已选检查点，防止把同一训练轨迹的多个检查点算作独立种子。少于三个有效配对种子会明确标记样本不足，即使达到三个也只给描述统计，不宣称统计显著。

选定检查点后另建 `split=test` 的计划，使用保留测试对手。测试结果不会再次选择模型。时间／分差／历史方案的收益应比较多个相同训练种子上的结果；同一个确定性对局的重复次数不算独立样本。

该入口是独立离线选模流程，没有自动嵌入训练主循环；可对定期保存的 checkpoint 导出后运行。接口验证和官方硬件时延资格仍使用原 `holoboxing validate`；本次没有将本地 5 ms 资格门槛擅自改成 10 ms。
