# 劈叉展示任务（SplitsShow）

任务 ID：`Mjlab-SplitsShow-Flat-MicroDuck-HD1910`

在已有劈叉技能 [splits-task.md](splits-task.md) 的基础上做一次完整展示：

```text
STAND → TWIST → PRESS → SPLIT →（保持 5 s）
      → PRESS → TWIST → STAND →（站稳 2 s）
```

固定 **左腿在前**，起身沿**原路返回**（SPLIT → PRESS → TWIST → STAND），最后稳稳站住。
复用原劈叉任务的全部执行器、传感器、终止、求解器与观测/动作契约
（61D 观测、14 维动作，`[twist(3), head(4), body(6)]` 命令块），几何姿态完全不变。

## 时间线

下劈沿用原劈叉动作的物理节奏（即原 9 s 周期内 0.15 / 0.32 / 0.48 对应的秒数），
因此动作速度与原任务一致；只因展示周期更长，归一化后的相位分数不同。
数据在 [splits_keyframes.py](../src/mjlab_microduck/tasks/splits_keyframes.py)。

周期 `SHOW_PERIOD_S = 17.0 s`，回合 `SHOW_EPISODE_LENGTH_S = 15.64 s`。

| 关键帧 | t (s) | phase = t/17 | 姿态 | 躯干高度 |
|---|---|---|---|---|
| STAND | 0.00 | 0.0000 | STAND | 0.115 m |
| TWIST | 1.35 | 0.0794 | TWIST | 0.114 m |
| PRESS | 2.88 | 0.1694 | PRESS | 0.095 m |
| SPLIT | 4.32 | 0.2541 | SPLIT | 0.063 m |
| SPLIT_HOLD | 9.32 | 0.5482 | SPLIT（保持 5 s） | 0.063 m |
| PRESS（返回） | 10.76 | 0.6329 | PRESS | 0.095 m |
| TWIST（返回） | 12.29 | 0.7229 | TWIST | 0.114 m |
| STAND（返回） | 13.64 | 0.8024 | STAND | 0.115 m |
| STAND_SETTLE | 15.64 | 0.9200 | STAND（站稳 2 s） | 0.115 m |

- 保持段用「SPLIT@0.2541 → SPLIT@0.5482」两个相同姿态的 waypoint 实现，
  不能依赖末端 clamp（否则从 4.32 s 就会往回走）。
- 相位分数严格递增；`p_end = 0.92 < 1`，且周期 17 s > 回合 15.64 s，相位时钟**不会回绕**到站立。
- 15.64 s / 0.02 s = 782 步（50 Hz 控制），整除。

## 奖励与窗口

复用原劈叉任务的奖励项与权重，仅把跟踪目标换成展示时间线、并为不同阶段加相位窗口：

| 奖励项 | 权重 | 说明 |
|---|---|---|
| `split_pose_legs` | 8.0 | 腿部 10 关节按展示时间线跟踪（含保持、返回、站稳） |
| `split_pose_neck` | 1.5 | 颈/头保持 HOME |
| `split_pose_l1` | 0.5 | 远距离恒定梯度 |
| `split_height` | 2.0 | 躯干高度 0.115→0.114→0.095→0.063→…→0.115 |
| `split_feet_grounded` | 1.5 | 相位 ≥ 0.1853（`0.35 × 9 / 17`）后双脚着地，起身/站稳同样奖励 |
| `split_hold_stillness` | 1.0 | 仅保持段 `[0.2541, 0.5482)` 奖励躯干/关节静止 |
| `split_stand_stillness` | 1.5 | 相位 ≥ 0.8024（回到站立）后奖励静止 → 最终站稳 2 s |
| `split_rise_bootstrap` | 0.5 | 起身窗口 `[0.5482, 0.8024)` 奖励躯干向上速度（限高 0.12 m、限速 0.08 m/s），给最难的起身段早期梯度 |
| `split_arrival_damper` | -0.03 | `body_ang_vel_at_height` 在站高附近抑制角速度；tilt gate 避免干扰下劈阶段 |
| `upright` | 1.0 | 躯干直立 |
| `body_ang_vel` | -0.05 | 躯干角动量 |
| `action_rate_l2` | -0.05 | 动作平滑 |
| `joint_torques_l2` | -2e-3 | 力矩正则 |
| `trunk_impact_penalty` / `head_impact_penalty` | -2.0 | 躯干/头部触地受力惩罚 |

> 符号约定：`body_ang_vel_at_height` 与所有 impact cost 返回**正的代价**，因此权重为负。
> 详见 `AGENTS.md`。

## 终止

沿用原劈叉任务并继承速度任务的判据：`trunk_ground_contact`、`head_ground_contact`
（躯干/头部触地即终止）、`nan_state`，以及通用 `fell_over`（倾角判定；
劈叉低位直立不会误触，与原劈叉一致）。

## 检查与预览

纯 CPU 几何检查（无需 GPU / mjlab），展示时间线只复用已验收的四个姿态：

```bash
python scripts/check_split_pose.py --timeline show
python scripts/check_split_pose.py --timeline show --render out/splits_show
```

本地可视化预览（同样只需 CPU MuJoCo + Pillow），默认使用展示周期与回合长度：

```bash
python scripts/preview_split_sim.py --mode kinematic --timeline show --out out/splits_show_preview
```

输出 `out/splits_show_preview/show_kinematic.gif` 与 `show_kinematic_strip.png`。
它播放的是**策略被要求学会的目标动作**，不是训练后的策略；`--mode pd` 同样可作开环对照。

## 训练

需要 Linux + NVIDIA GPU，或配置了 GPU 的 WSL2（BAM 要求 Python < 3.13，训练需要 CUDA）：

```bash
cd software/training
uv sync --frozen --python 3.12

# 先用少量环境检查链路
uv run --frozen train Mjlab-SplitsShow-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.logger tensorboard

# 正式训练
uv run --frozen train Mjlab-SplitsShow-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 --agent.max-iterations 6000 \
  --agent.logger tensorboard
```

日志目录：`logs/rsl_rl/microduck_splits_show/`。

TensorBoard 重点：

- `Episode_Reward/split_pose_legs` 上升且**起身段不塌**；
- `split_height` 跟随 0.115 → 0.063 → 0.115；
- `split_hold_stillness` 只在保持段非零，`split_stand_stillness` 在末尾非零；
- 所有惩罚项 ≤ 0；回合约 15.6 s。

测试（在 Linux/WSL 的 GPU 环境）：

```bash
uv run --frozen --with pytest python -m pytest -q \
  tests/test_splits_show_cfg.py tests/test_split_geometry.py
```

## 运行时接口

与所有 microduck 任务相同，`twist` 命令槽被复用为**可部署的相位时钟**：

```text
cmd_twist = [cos(2πp), sin(2πp), 0]
```

`p` 从 0 线性增长到 1，周期 **17 s**（大于 15.64 s 回合长度，因此不会绕回站立）。
按下“劈叉展示”按钮时把 `p` 置 0 并开始推进：

- `p ≥ 0.2541`：进入劈叉；
- `p = 0.5482`：结束 5 s 保持，开始起身；
- `p = 0.8024`：回到站立位；
- `p → 0.92`：站稳段结束。

观测与动作维度与其它 microduck 策略一致（61D 观测、14 维动作）。
模型导出仍用 `scripts/export.py`，输出 `policy.onnx`（含观测归一化）。

## 导出与部署

导出命令（与其它任务相同，按任务 ID 取）：

```bash
uv run --frozen export Mjlab-SplitsShow-Flat-MicroDuck-HD1910 \
  --checkpoint logs/rsl_rl/microduck_splits_show/<run>/model_XXXX.pt \
  --output exports/splits_show/policy.onnx
```

> 注意：手机控制台的 Studio 目前只有 7 种固定动作槽位
> （walk / sitstand / ground_pick / kick_left / kick_right / roulade / stand），
> **没有 splits 槽位**，`_recommended_slot` 对 `Mjlab-Splits-*` / `Mjlab-SplitsShow-*`
> 返回 `None`，因此即使 `manifest.json` / `deployment-contract.json` 填写正确也无法导入。
> 需要先在 Studio 增加 splits 槽位才能部署到实机。详见
> [spacehpc-splits-tutorial.md](spacehpc-splits-tutorial.md) 第 9 节。
