当前实际运行方案：stance-alignment-v10 第二轮

本目录保存产生 60 秒 demo、31.170 秒右直拳合法命中躯干的实际运行代码。
源码保持验证时内容，snapshot.json 记录文件 SHA256、外部动作指纹和结果摘要。
这是运行快照，不是新的通用训练框架，也不是可提交的独立 ONNX。

执行链
测量双方状态 → 骨盆/双脚联合对齐规则 → 有状态动作库调度 → 五点 C225 → 冻结下层。
动作类别：前进、转体、直拳、勾拳。转体分 5°、10°、20°；近身有短步及收稳保持。
近身起拳门限为骨盆对应拳路偏角8°、双脚站位偏角12°。
对齐后，骨盆水平距离(0.50,0.60]m使用直拳，<=0.50m使用勾拳，>0.60m接近。
选择仅发生在整组动作结束时；出拳过程中仍可出现超过门限的偏角。

提交内容
- stance_alignment_control.py：实测骨盆/脚步偏差、分区门限、参考跟踪偏差补偿及离散选角。
- evaluate_alignment.py：当前实际使用的无训练对战入口、短步/收稳/动作拼接、C225编码、裁判接触采集。
- analyze_alignment.py：当前测试与旧基线数据的朝向、冲量、得分对照。它读取基线数据，不包含旧策略代码。
- render_demo.py：从已保存的真实物理轨迹生成完整视频、慢放与关键帧。
- check_alignment.py：实际使用的52种几何/决策情景、全局坐标变换不变性和跟踪偏差检查。
- alignment.json、candidate_50hz下的规则/特征/校验文件：运行所读配置。
- 仓库环境适配的最小依赖：mujoco_env.py中的复位前状态采集接口，以及official_referee.py中的合法躯干接触统计。

没有包含旧强化学习实验、未完成行为克隆/ONNX尝试、14点肘部扩展、过渡方案、
旧动作构建历史、模型权重、动作NPZ或视频。现有公共下层、场景及裁判代码沿用仓库。

使用已验收动作文件
必须另行提供 candidate_50hz/motions 下5个基础动作、selector.pt、selector-weights.json，
以及 approach 下4种短步和4种小转体NPZ。基础动作以candidate_50hz/integrity.json校验；
approach文件指纹见snapshot.json。不得把缺失的文件换成同名旧动作。
selector.pt是实际入口仍读取的11→4分类器，仅用于记录原始建议；最终选招被对齐规则决定。
保留它及相关读取逻辑是为了保持本次实际运行代码，不代表它是最终执行决策或1097→225模型。

现有服务器运行目录：
  /home/zhanghan/projects/HBot-stance-alignment-v10-20261106
该目录已有完整私有运行资产，最终结果eval-v2，录像demo-v2。
可将本目录文件复制到新的完整运行目录，与candidate_50hz、approach并列；
并通过PYTHONPATH或目录链接让本仓库holoboxing、holomotion包可导入。
原始脚本保留了本次环境的资源路径、GPU UUID和蓝方模型路径，迁移到另一机器前须检查这些设置。
不要覆盖正在使用的代码或既有eval-v2/demo-v2成果。

已验证环境
Linux，Python环境 /home/zhanghan/envs/node04/holoboxing/bin/python；
Torch 2.6.0+cu124、ONNX 1.18.0、ONNX Runtime 1.22.0、MuJoCo 3.5.0。
物理GPU1 RTX3090；脚本以CUDA_VISIBLE_DEVICES绑定该卡，因此策略看到cuda:0。
视频辅助依赖路径 /home/zhanghan/tools/hbot-video。
竞赛资产 /home/zhanghan/datasets/hbot-assets；冻结下层 /home/zhanghan/datasets/holomotion-v2。
冻结蓝方是入口中指定的seed44/model_1536.pt；指纹见snapshot.json。
原生CPU物理，控制50Hz、物理200Hz；标准五点目标，没有额外肘肩目标。

在完整运行目录执行（Bash）
  /home/zhanghan/envs/node04/holoboxing/bin/python evaluate_alignment.py --adaptive --output eval-new

eval-new须为不存在的新目录。--adaptive必须保留，它会加载短步、小转体和settle_guard。
此入口固定seed273、单环境、最长60秒，约205秒墙钟时间；不会训练。
输出report.json、states.npz、decisions.json、metrics.json、contacts.json。

检查及分析
  python check_alignment.py
  python analyze_alignment.py eval-v2

check_alignment.py需要上述straight_lr.npz，生成control-preflight.json。
analyze_alignment.py还读取相邻HBot-standard-five-range060-20261106目录中的demo与impulse-check基线数据。
若只运行新对战，不需要该基线目录。

录像
  python render_demo.py

只适用于对应的eval-v2；脚本固定读取eval-v2/states.npz、固定写demo-v2，拒绝覆盖。
当前比分字幕、31.17秒命中提示、5.191 N·s数值与慢放时段也固定对应本次结果。
不能仅换轨迹目录就用它给另一场比赛贴相同字幕；新场次应根据实际contacts.json生成字幕。

配置注意
candidate_50hz/rules.json保留旧30°描述；本次实际朝向由alignment.json和choose()决定。
距离、短步参数仍读取rules.json。geometry()中的desired_heading_rad是诊断字段；第二轮
choose()实际使用0.75×骨盆偏角+0.25×双脚偏角及跟踪差补偿。
20°补偿和25°参考领先预算在代码内；25°是离散选角前的预算，不是物理姿态硬限位。
不要只改diagnostic字段所用foot_target_weight就认为执行权重已经改变。

证据边界
当前单场60秒测试：双方无倒地、无非法物理状态，红方1次有效击打，蓝方0次。
31.170秒红方右直拳接触躯干，20ms法向冲量5.191330384949788 N·s，超过5 N·s门槛。
36次攻击启动全部满足门限；骨盆/双脚起拳偏角中位数约4.52°/4.81°。
视频为保存物理轨迹的渲染，非参考动作播放；完整解码和关键帧检查已通过。
这不是线上平台资格结果或跨种子鲁棒性结论。没有启动新训练，也没有完成独立ONNX导出。
