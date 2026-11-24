# 劈叉（竖叉）任务

任务 ID：`Mjlab-Splits-Flat-MicroDuck-HD1910`

固定 **左腿在前**，从站立依次经过扭腰、下压，最后双腿伸展、脚/小腿着地并保持。
执行器与其它 HD1910 任务完全一致（`FeetechBamActuatorCfg` + `hd1910/1910_m6.json`）。

> 需要“劈叉 → 保持 5 s → 原路起身站稳”的完整展示，见
> [splits-show-task.md](splits-show-task.md)（任务 `Mjlab-SplitsShow-Flat-MicroDuck-HD1910`）。

## 关键帧

关键帧数据在 [splits_keyframes.py](../src/mjlab_microduck/tasks/splits_keyframes.py)：

| 相位 | 名称 | 躯干高度 | 姿态 |
|---|---|---|---|
| 0.00 | STAND | 0.115 m | HOME 站立 |
| 0.15 | TWIST | 0.114 m | 髋 yaw 扭转、髋 pitch 前后分开 |
| 0.32 | PRESS | 0.095 m | 屈髋下压 |
| 0.48 | SPLIT | 0.063 m | 左腿约 89°；后腿大腿根向后旋转、膝盖尽量伸直、小腿平贴地面，脚背朝下、脚尖朝后（脚掌朝上），双脚着地，左小腿贴地 |

相位 0.48 之后目标保持不变，因此剩余时间都在保持劈叉。

CPU 可行性检查（不需要 GPU / mjlab）：

```bash
python scripts/check_split_pose.py
python scripts/check_split_pose.py --render out/
```

脚本会报告每个关键帧的最低点与地面接触，并在脚离地超过 3 mm 或躯干/头部触地时返回非零。
当前结果：SPLIT 相位双脚离地 ≤ 2 mm、左小腿贴地，躯干离地 19 mm、头部 83 mm。

本地可视化预览（同样不需要 GPU / mjlab，只需要 CPU MuJoCo + Pillow）：

```bash
python scripts/preview_split_sim.py --mode kinematic --out out/split_preview
```

输出 `out/split_preview/split_kinematic.gif`（8 s 动画）和 `split_kinematic_strip.png`（8 帧拼图）。
它按 9 s 周期相位时钟播放关键帧目标，展示**策略被要求学会的动作**，不是训练后的策略。

`--mode pd` 用普通位置伺服（HD-1910 力矩上限）闭环跑同一轨迹作为对照：关节跟踪误差 < 0.01 rad，
但躯干会前倾落地（约 0.038 m 触发躯干接触）。这正说明开环跟踪不足以完成劈叉，必须靠 RL 学平衡。

> 几何限制：Microduck 腿短、头部重，**双腿完全水平且双脚同时着地需要躯干触地**，无法实现。
> 因此最终姿态是“左腿接近水平、后腿大腿根向后旋转、膝盖尽量伸直、小腿平贴地面”的最深稳定折中：
> 该腿的膝部连杆本身最直也只能到约 37°（膝角约 1.32 rad / 75.6°），所以后膝保持在这个最直角度；
> 踝关节前屈接近极限（-89.4°），使脚背（脚的上表面）朝下、脚尖朝后、脚掌朝上，像人劈竖叉后腿的脚。
> 髋关节离地约 46 mm，而后腿大腿只有约 42 mm，所以大腿下缘最低约 10 mm，无法完全贴地。
> 膝盖/小腿/脚背允许着地，躯干和头不允许（训练中会终止）。

## 训练

需要 Linux + NVIDIA GPU，或配置了 GPU 的 WSL2（Windows 原生 Python 3.14 不支持本项目依赖，
BAM 要求 Python < 3.13，且训练需要 CUDA）：

```bash
cd software/training
uv sync --frozen --python 3.12

uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 --agent.max-iterations 6000 \
  --agent.logger tensorboard
```

训练中重点看 `Episode_Reward/split_pose_legs` 是否上升、`split_height` 是否跟随，
以及所有惩罚项是否 ≤ 0（符号约定见 `AGENTS.md`）。

先用少量环境检查链路：

```bash
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.logger tensorboard
```

日志目录：`logs/rsl_rl/microduck_splits/`。

## 奖励与终止

- `split_pose_legs`（8.0）：腿部 10 个关节按相位跟踪关键帧（Gaussian，std 0.25）。
- `split_pose_neck`（1.5）：颈/头保持 HOME。
- `split_pose_l1`（0.5）：远距离时的恒定梯度。
- `split_height`（2.0）：躯干高度跟踪 0.115 → 0.114 → 0.095 → 0.063。
- `split_feet_grounded`（1.5）：相位 ≥ 0.35 后奖励双脚触地。
- `split_hold_stillness`（1.0）：相位 ≥ 0.48 后奖励躯干/关节静止。
- `upright`（1.0）、`body_ang_vel`（-0.05）、`action_rate_l2`（-0.05）、`joint_torques_l2`（-2e-3）。
- `trunk_impact_penalty` / `head_impact_penalty`（-2.0）：躯干/头部触地受力惩罚。
- 终止：`trunk_ground_contact`、`head_ground_contact`、`nan_state`。

## 运行时接口

`twist` 命令槽被复用为相位时钟，与 ground-pick/sit-stand 相同的部署方式：

```text
cmd_twist = [cos(2πp), sin(2πp), 0]
```

`p` 从 0 线性增长到 1，周期 9 s（大于 8 s 回合长度，因此不会绕回站立）。
按下“劈叉”按钮时把 `p` 置 0 并开始推进；`p ≥ 0.48` 即进入保持段。
观测与动作维度与其它 microduck 策略一致（61D 观测、14 维动作）。
