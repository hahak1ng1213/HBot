ts_1.0 — student_1.0 教师蒸馏版本（2026-11-07）

交付模型：artifacts/student_1.0.onnx
训练检查点：artifacts/student_1.0.pt
原始实验：student-enriched-0/latest.pt，epoch 88。
重命名不改变权重或 ONNX 图；模型 SHA256：
1ee3d8f0c9dccaa25f3dcb0e81902104fee252f52667b68e974d7a3f2ead87c9
检查点 SHA256：
842cdc018a0d1ffd0ea5518cedfa20812cced5511a1a59ba2ddaa6361de5be4e

输入 obs float32[1,1097]，输出 target_pose float32[1,225]。
输出是当前完整骨盆坐标系中的未来五帧、五个控制点，每点位置3维与旋转X/Z轴6维。
模型无外部状态，权重及归一化内置，推理不需要教师或动作库；只提交 ONNX 即可。
标准 MLP 1095→512→512→256→225，ELU、当前姿态残差与旋转正交化。
1097维公开输入中的最后2维比赛上下文未用于网络计算。

本目录保留核心采集、训练、导出、实际仿真、视频渲染与必要数值/运动诊断代码。
未纳入未采用的 retrieval、temporal、lead、fast-export、DAgger 调度实验，
一次性诊断/测时代码、进程控制代码、原始训练样本、缓存和日志。
collect_student.py 的教师混合采集能力仍保留；本版最后一次训练只使用教师执行数据。
历史初始化来自兼容学生权重，不能据此声称整个权重历史从未使用过其他数据。

验证结果（冻结蓝方 seed44，原生 MuJoCo，单场60秒）：
  随机种子273：学生12:0，教师7:0；用于选择。
  随机种子284：学生12:0，教师17:0；用于选择。
  随机种子371：学生2:0，教师7:0；模型固定后的留出测试。
  随机种子577：学生7:0，教师3:0；模型固定后的留出测试。
四场学生均无倒地、无非法帧。3000个实际输入 CPU/CUDA 数值检查零失败，
20次重复推理一致。详见 reports/validation-report.json、numerical-parity.json。
不能将四场结果等同于普遍达到教师能力或正式比赛资格通过。
尚未上传比赛平台。用户要求暂不测时，本分支未继续测时或性能优化。

源码与外部运行依赖

bundle/state_aware_control.py 与13个 bundle/motions/*.npz 是本版实际使用的教师。
bundle/manifest.json 保留原始完整指纹，包括运行库、对手模型和来源路径。
为避免重复上传整套运行库及其中的旧实验/测试，bundle/runtime 不纳入本次提交。
现有仓库根目录的运行库与冻结快照不是逐文件完全一致；不能直接替换后跳过哈希校验。
服务器上的完整原始 bundle 是后续运行的首选，保留在：
  /home/zhanghan/projects/HBot-student-submission-20261107/bundle
另外仍需要服务器上的机器人资产、冻结下层、冻结蓝方和原始训练数据。
GitHub 中的 ONNX 可独立推理，但仅克隆仓库不等于拥有完整仿真/训练环境。

在已有服务器执行（Bash，先检查空闲GPU，输出目录必须是新的）：

  cd /path/to/HBot/examples/ts_1.0
  PY=/home/zhanghan/envs/node04/holoboxing/bin/python
  SOURCE=/home/zhanghan/projects/HBot-student-submission-20261107
  BUNDLE="$SOURCE/bundle"
  # 先用 nvidia-smi 确认空闲卡，再设置 CUDA_VISIBLE_DEVICES 为所选 UUID。
  # student_common.py 默认是当时的 GPU0，不能假定它现在仍空闲。

独立学生60秒评测（不训练，teacher-fraction 必须为0）：
  "$PY" collect_student.py --bundle "$BUNDLE" --output runs/student-1.0-seed273 \
    --worlds 1 --seconds 60 --seed 273 --native-reset \
    --onnx artifacts/student_1.0.onnx --teacher-fraction 0 --evaluate

同条件教师基线：
  "$PY" collect_student.py --bundle "$BUNDLE" --output runs/teacher-seed273 \
    --worlds 1 --seconds 60 --seed 273 --native-reset --teacher-fraction 1 --evaluate

保存真实轨迹的视频与物理诊断：
  "$PY" render_student.py --bundle "$BUNDLE" --source runs/student-1.0-seed273 \
    --output runs/student-1.0-demo
  "$PY" audit_motion.py --bundle "$BUNDLE" --source runs/student-1.0-seed273

从已交付权重继续拟合现有教师数据（仅在决定继续训练后运行）：
  "$PY" train_student.py --bundle "$BUNDLE" \
    --data "$SOURCE/teacher-general" "$SOURCE/teacher-close" \
      "$SOURCE/teacher-native" "$SOURCE/teacher-enrichment-1" \
    --checkpoint artifacts/student_1.0.pt --output runs/student-next --epochs 200 --seed 71401
此命令是继续训练，不是从头复现本版。--checkpoint 只恢复权重、不恢复优化器。
原训练先从旧项目 student-2/best.pt warmstart 得到 student-0/best.pt，
再增加 teacher-enrichment-1 拟合至 epoch88；确切样本指纹见 training-status.json。
划分采用 world 6/7 验证，共67200训练行、22500验证行。

重新导出（默认输出名 student.onnx；本次已验收文件发布名为 student_1.0.onnx）：
  "$PY" export_student.py --bundle "$BUNDLE" --checkpoint artifacts/student_1.0.pt \
    --data "$SOURCE/teacher-general" --output runs/reexport
不传 --measure-upper-timing。重新导出的字节未必与已验收文件完全相同，
请优先使用 artifacts/student_1.0.onnx；任何新模型需要重新做实际闭环对战。

如需在同一服务器使用本目录的 bundle：
  cp -a "$SOURCE/bundle/runtime" bundle/runtime
  "$PY" -c 'from student_common import setup; setup("bundle"); print("bundle verified")'
这只复制已有运行库，setup 同时校验动作、代码及外部蓝方指纹。
迁移服务器要先恢复这些依赖；不要为绕过错误而随意改 manifest。

保留的 demo

demos/student/motion.mp4：最终 student_1.0，seed273，60秒12:0，独立 ONNX。
demos/teacher-accepted/motion.mp4：用户认可的教师收稳调整版60秒对战，
  来源 eval-retract-20261107-targeted-settle；它早于最终学生冻结包，不能当作
  reports 中 seed273 教师7:0基线的视频，也不能声称与最终教师文件逐字节一致。
demos/straight-retraction-comparison/motion.mp4：8秒连续直拳回收诊断，
  左旧右新，实际物理、正常速度；用于保留用户认可的拳路目标，不是完整对战。
每段均带 overview.jpg 和 video-report.json，来源轨迹与视频指纹可核对。

已知问题与诊断注意

最终学生 demo 在20–30秒约4次明显伸拳/+1分，30–40秒0次/+0，
40–50秒15次/+6，50–60秒14次/+1。伸拳数是轨迹估算，不是裁判命中数。
20–40秒距离中位数66.4cm，19.44秒超过教师60cm起拳门槛；朝向通常合格。
建议继续研究追近、开始/维持攻击的不同条件、拳套到胸部可达性及学生模仿差异。
这里只分析了问题，没有修改策略、重训或验证修复效果。
reports/quiet-window-analysis.json 的算法为双拳前伸峰：prominence>=0.12m、
reach>=0.30m、间距>=0.16s；按原始保存状态计算，不加0.75m近身过滤。
audit_motion.py 的近身统计另有 distance<0.75m 过滤，因此计数不必一致。

重要：collect_student.py 在纯学生评测时仍计算教师标签，但完全不参与执行。
metrics.json 的 skills 是影子教师建议，不是学生的离散动作状态。
影子教师参考未被执行，跟踪误差及 recover 标签不能直接代表学生为何收稳。
两场独立教师/学生对战同一秒的世界状态也不相同，不能当严格反事实对照。

完整上下文、服务器信息和后续交接见仓库根目录 HBot_ts_1.0_交接_20261107.txt。
